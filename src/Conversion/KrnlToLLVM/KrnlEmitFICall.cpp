/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlEmitFICall.cpp - Lower KrnlInjectFICallOp ----------------===//
//
// Copyright 2022 Udit K. Agarwal.
//
// =============================================================================
//
// This file lowers the KrnlInjectFICallOp operator.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"
#include "src/Dialect/Krnl/KrnlHelper.hpp"

#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlInjectFICallOpLowering : public ConversionPattern {
public:
  explicit KrnlInjectFICallOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlInjectFICallOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto FITensorOp = cast<KrnlInjectFICallOp>(op);
    MLIRContext *context = FITensorOp.getContext();
    Location loc = FITensorOp.getLoc();
    KrnlPrintTensorOpAdaptor operandAdaptor(operands);

    StringRef msg = FITensorOp.msg();
    Value input = operandAdaptor.input();
    assert(input.getType().isa<LLVM::LLVMStructType>() &&
           "expecting LLVMStructType");

    ModuleOp module = FITensorOp->getParentOfType<ModuleOp>();
    const auto &apiRegistry = RuntimeAPIRegistry::build(module, rewriter);


    // Get a symbol reference to the runtime function to use, creating one if
    // necessary.
    auto int64Ty = IntegerType::get(context, 64);
    auto memRefTy = input.getType().dyn_cast<LLVM::LLVMStructType>();
    auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
    auto memRefRankVal = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(memRefRank));

    // Create a Tensor out of the data pointer. The idea is to get dynamic shape from tensor
    // at runtime in the LLTFI's FI library.
    Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

    Value owning = rewriter.create<LLVM::ConstantOp>(
      loc, int64Ty, rewriter.getI64IntegerAttr(0));

    // Get Allocated pointer and convert it to i8*
    Value outMemRefAllocatedPtr =
      rewriter.create<LLVM::ExtractValueOp>(loc, memRefTy.getBody()[0],
          input, rewriter.getArrayAttr({rewriter.getI64IntegerAttr(0)}));

    outMemRefAllocatedPtr = rewriter.create<LLVM::BitcastOp>(loc,
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
      outMemRefAllocatedPtr);

    // Get Aligned pointer and convert it to i8*
    Value outMemRefAlignedPtr =
      rewriter.create<LLVM::ExtractValueOp>(loc, memRefTy.getBody()[1],
          input, rewriter.getArrayAttr({rewriter.getI64IntegerAttr(1)}));

    outMemRefAlignedPtr = rewriter.create<LLVM::BitcastOp>(loc,
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
      outMemRefAlignedPtr);

    // Set the data into the Tensor
    RuntimeAPI::callApi(rewriter, loc, apiRegistry, RuntimeAPI::API::SET_DATA,
      {omTensor, owning, outMemRefAllocatedPtr, outMemRefAlignedPtr});

    Type elemTy =
      memRefTy.getBody()[0].cast<LLVM::LLVMPointerType>().getElementType();

    onnx::TensorProto::DataType onnxTy = krnl::mlirTypeToOnnxType(elemTy);
    auto onnxTyVal = rewriter.create<LLVM::ConstantOp>(
      loc, int64Ty, rewriter.getI64IntegerAttr(onnxTy));

    // Set the data type of Tensor.
    RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::SET_DATA_TYPE, {omTensor, onnxTyVal});

    // Get Rank and Data Shape
    int64_t rank = krnl::getRankFromMemRefType(memRefTy);
    Value sizesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::GET_DATA_SHAPE, {omTensor});
    Value stridesArrayPtr = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
      RuntimeAPI::API::GET_DATA_STRIDES, {omTensor});

    for (decltype(rank) i = 0; i < rank; i++) {
      auto dimIdx = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(i));

      // Transfer size of dimension from memref to dynamic memref.
      auto dimSize = rewriter.create<LLVM::ExtractValueOp>(loc, int64Ty,
          input,
          rewriter.getArrayAttr(
              {rewriter.getI64IntegerAttr(3), rewriter.getI64IntegerAttr(i)}));
      auto dimSizePtr =
          rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(int64Ty),
              sizesArrayPtr, ArrayRef<Value>({dimIdx}));
      rewriter.create<LLVM::StoreOp>(loc, dimSize, dimSizePtr);

      // Transfer stride of dimension from memref to dynamic memref.
      auto dimStride = rewriter.create<LLVM::ExtractValueOp>(loc, int64Ty,
                      input, rewriter.getArrayAttr(
          {rewriter.getI64IntegerAttr(4), rewriter.getI64IntegerAttr(i)}));
      auto dimStridePtr =
          rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(int64Ty),
          stridesArrayPtr, ArrayRef<Value>({dimIdx}));
      rewriter.create<LLVM::StoreOp>(loc, dimStride, dimStridePtr);
    }

    // Get/Inject call to LLTFI's injectFault() function
    auto injectFaultRef = getOrInsertInjectFault(rewriter, module);

    // Convert operator name to a Global String
    LLVM::GlobalOp formatSpec = getOrCreateGlobalString(msg, loc, rewriter,
        module, static_cast<LLVMTypeConverter *>(getTypeConverter()));
    Value formatSpecPtr = getPtrToGlobalString(formatSpec, loc, rewriter);

    // Convert Data pointer to i8* type
    outMemRefAlignedPtr =
      rewriter.create<LLVM::ExtractValueOp>(loc, memRefTy.getBody()[1],
          input, rewriter.getArrayAttr({rewriter.getI64IntegerAttr(1)}));

    outMemRefAlignedPtr = rewriter.create<LLVM::BitcastOp>(loc,
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
      outMemRefAlignedPtr);

    // COnvert rank to a constant
    Value rankConst = rewriter.create<LLVM::ConstantOp>(
      loc, int64Ty, rewriter.getI64IntegerAttr(rank));

    // Convert DataShapePointer and Strides to i8*
    sizesArrayPtr = rewriter.create<LLVM::BitcastOp>(loc,
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
      sizesArrayPtr);

    stridesArrayPtr = rewriter.create<LLVM::BitcastOp>(loc,
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
      stridesArrayPtr);

    // Inject call to LLTFI's Inject Fault function.
    rewriter.create<func::CallOp>(loc, injectFaultRef, ArrayRef<Type>({}),
        ArrayRef<Value>({formatSpecPtr, outMemRefAlignedPtr, rankConst, sizesArrayPtr,
                         stridesArrayPtr}));

    rewriter.eraseOp(op);
    return success();
  }

private:
  static FlatSymbolRefAttr getOrInsertInjectFault(PatternRewriter &rewriter,
         ModuleOp module) {

    auto fIFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("LLTFIInjectFault");
    MLIRContext *ctx = rewriter.getContext();

    if (!fIFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto voidType = LLVM::LLVMVoidType::get(ctx);
      Type i8Type = IntegerType::get(ctx, 8);
      Type i64Type = IntegerType::get(ctx, 64);
      Type i8PtrType = LLVM::LLVMPointerType::get(i8Type);
      fIFunc =
          rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), "LLTFIInjectFault",
              LLVM::LLVMFunctionType::get(voidType, {i8PtrType,
              LLVM::LLVMPointerType::get(i8Type), i64Type,
              LLVM::LLVMPointerType::get(i8Type),
              LLVM::LLVMPointerType::get(i8Type)},
              /*isVarArg=*/false));
    }
    return SymbolRefAttr::get(ctx, "LLTFIInjectFault");
  }
};

void populateLoweringKrnlInjectFICallOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlInjectFICallOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
