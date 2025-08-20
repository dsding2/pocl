// LLVM module pass that adds a constant offset to all addressess that could
// access SVM regions.
//
// Copyright (c) 2024 Pekka Jääskeläinen / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/* An LLVM pass to adjust LLVM IR's memory accessess with a fixed offset.

   The adjustment is needed in case the host's SVM region's start address
   differs from the device's.

   Input: A non-adjusted kernel with all global pointers assumed to be
   pre-adjusted by the runtime to point to the SVM region of the targeted
   device.

   The pass works as follows with the general principle that the pointer
   addressess are adjusted to the correct offset at the point of a memory
   access. Due to generic pointers it is not always possible to figure out
   if the address space of a pointer is global or not, thus we must ensure
   all pointers can be adjusted at their usage time, thus they have to be
   negatively adjusted at the pointer creation or "import time".

   This means that all pointers that are created by

   - allocas,
   - when taking an address of a global variable or are
   - input to the kernel as arguments

   are negatively adjusted so we can adjust them back when accessing the
   memory. It leans heavily to compiler to remove the unnecessary
   negative/positive adjustment pairs.

   Pointer arguments to calls to defined functions are not adjusted, but
   the functions itself are handled separately. Arguments to undefined
   functions are assumed to be builtin functions which expect valid fixed
   pointers, thus they are adjusted at the call site.

   TODO: Indirect accesses? They are pointers loaded from another buffer
   or such. The pointers are global SVM so they should be fixed as well.
*/

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/ADT/Statistic.h>
// #include <llvm/Analysis/LoopInfo.h>
// #include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
POP_COMPILER_DIAGS

#include "LLVMUtils.h"
#include "AddIndirection.hh"
#include "WorkitemHandlerChooser.h"
#include "KernelCompilerUtils.h"


#include <iostream>
#include <set>
#include <string>
#include <queue>

//#include "pocl_llvm_api.h"

#define PASS_NAME "add-indirection"
#define PASS_CLASS pocl::AddIndirection
#define PASS_DESC "Adds indirection"

using namespace llvm;

namespace pocl {

class AddIndirectionImpl : public pocl::WorkitemHandler {
public:
  AddIndirectionImpl(llvm::TargetTransformInfo &TTI)
      : WorkitemHandler(), TTI(TTI) {}
  bool runOnFunction(llvm::Function &Func) {
    M = Func.getParent();
    F = &Func;

    const DataLayout &DL = M->getDataLayout();

    Initialize(cast<Kernel>(&Func));

    int vectorization_dim = 2;
    IterSize = WGLocalSizeZ;
    if (WGLocalSizeX >= WGLocalSizeY && WGLocalSizeX >= WGLocalSizeZ) {
      IterSize = WGLocalSizeX;
      vectorization_dim = 0;
    } else if (WGLocalSizeY >= WGLocalSizeX && WGLocalSizeY >= WGLocalSizeZ) {
      vectorization_dim = 1;
      IterSize = WGLocalSizeY;
    }

    std::queue<Value*> worklist;
    std::set<Instruction*> dependents;

    unsigned scalarBitWidth = 0;

    for (Instruction &I : instructions(F)) {
      llvm::CallInst* Call = dyn_cast<llvm::CallInst>(&I);
      if (Call == nullptr)
        continue;

      if (isCompilerExpandableWIFunctionCall(*Call)) {
        auto Callee = Call->getCalledFunction();
        int Dim =
            cast<llvm::ConstantInt>(Call->getArgOperand(0))->getZExtValue();
        if (Callee->getName() == GID_BUILTIN_NAME && Dim == vectorization_dim) {
          scalarBitWidth = DL.getTypeSizeInBits(Call->getType());
        } 
      }
    }

    if (scalarBitWidth == 0) {
      return false;
    }
    unsigned regWidth = SizeTWidth;
    GangSize = regWidth / scalarBitWidth;

    Function *Helper = AddIndirection();

    Type *VecType = VectorType::get(ST, GangSize, false);
    GlobalIdVecs = {
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(0), VecType)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(1), VecType)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(2), VecType))};
    
    for (Instruction &I : instructions(Helper)) {
      llvm::CallInst* Call = dyn_cast<llvm::CallInst>(&I);
      if (Call == nullptr)
        continue;

      if (isCompilerExpandableWIFunctionCall(*Call)) {
        auto Callee = Call->getCalledFunction();
        int Dim =
            cast<llvm::ConstantInt>(Call->getArgOperand(0))->getZExtValue();
        if (Callee->getName() == GID_BUILTIN_NAME && Dim == vectorization_dim) {
          dependents.insert(Call);
        } 
      }
    }
    
    std::vector<Instruction *> UserList;
    for (Instruction* I : dependents) {
      for (User *U : I->users()) {
        Instruction *DepInst = dyn_cast<llvm::Instruction>(U);

        if (DepInst) {
          UserList.push_back(DepInst);
        }
      }
    }
    for (Instruction* I : dependents) {
      for (Instruction *DepInst : UserList) {
        vectorizeInstruction(DepInst, I, std::prev(Helper->arg_end()));
      }
    }
    
    return true;
  }

private:
  llvm::Module *M;
  llvm::Function *F;
  llvm::TargetTransformInfo &TTI;

  std::array<llvm::GlobalVariable *, 3> GlobalIdVecs;
  std::array<llvm::GlobalVariable *, 3> LocalIdVecs;
  int GangSize;
  int IterSize;

  std::vector<llvm::Instruction *> NewInstructions;

  Function* AddIndirection() {
    // Add indirection
    std::vector<Type*> ParamTypes;
    for (auto &Arg : F->args()) {
      ParamTypes.push_back(Arg.getType());
    }

    Type *VecType = VectorType::get(ST, GangSize, false);
    ParamTypes.push_back(VecType);

    FunctionType *FTy = FunctionType::get(F->getReturnType(), ParamTypes, F->isVarArg());
    Function *Helper = Function::Create(
        FTy,
        Function::InternalLinkage,
        F->getName().str() + "_helper",
        M
    );

    F->setName("_pocl_kernel_" + F->getName().str() + "_workgroup");

    ValueToValueMapTy VMap;
    auto NewArgIt = Helper->arg_begin();
    for (auto &OldArg : F->args()) {
        VMap[&OldArg] = &(*NewArgIt);
        // (*NewArgIt).setName(OldArg.getName());
        ++NewArgIt;
    }

    Helper->splice(Helper->begin(), F);

    for (BasicBlock &BB : *Helper) {
        for (Instruction &I : BB)
            RemapInstruction(&I, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    }

    std::vector<Value*> Args;
    for (auto &Arg : F->args()) {
        Args.push_back(&Arg);
    }
    Args.push_back(nullptr);


    // wrap in loop
    BasicBlock *entry;
    if (F->empty()) {
        entry = BasicBlock::Create(F->getContext(), "new_entry", F);
    } else {
        entry = &(F->getEntryBlock());
    }

    IRBuilder<> Builder(entry);
    std::vector<Constant*> Indices;
    for (unsigned k = 0; k < GangSize; ++k) {
        Indices.push_back(ConstantInt::get(ST, k));
    }
    Constant *ConstVec = ConstantVector::get(Indices);

    BasicBlock *EntryBB = Builder.GetInsertBlock();
    Value *N = ConstantInt::get(ST, IterSize); 

    // TODO: fix this to actually work
    BasicBlock *LoopBB = BasicBlock::Create(F->getContext(), "loop", F);
    BasicBlock *AfterBB = BasicBlock::Create(F->getContext(), "afterloop", F);

    Builder.CreateBr(LoopBB);

    Builder.SetInsertPoint(LoopBB);

    PHINode *IV = Builder.CreatePHI(ST, 2, "i");
    IV->addIncoming(ConstantInt::get(ST, 0), EntryBB);

    Value *Splat = Builder.CreateVectorSplat(GangSize, IV);
    Value *Result = Builder.CreateAdd(Splat, ConstVec, "i_plus_offset");
    Args[Args.size()-1] = Result;
    Builder.CreateCall(Helper, Args);

    Value *NextIV = Builder.CreateAdd(IV, ConstantInt::get(ST, GangSize), "i.next");

    Value *Cond = Builder.CreateICmpSLT(NextIV, N);
    Builder.CreateCondBr(Cond, LoopBB, AfterBB);
    IV->addIncoming(NextIV, LoopBB);

    Builder.SetInsertPoint(AfterBB);
    Builder.CreateRetVoid();
    return Helper;
  }

  void vectorizeInstruction(llvm::Instruction* I, llvm::Value* oldVal, llvm::Value* newVal) {
    Instruction *nextReplacement = nullptr;
    BasicBlock *tempBB = nullptr;
    if (isVectorizableInstruction(I)) {
      tempBB = vectorizedReplace(I, oldVal, newVal, &nextReplacement);
    } else {
      tempBB = unvectorizedReplace(I, oldVal, newVal, &nextReplacement);
    }
    // assert(nextReplacement);
    std::vector<Instruction *> UserList;
    for (User *U : I->users()) {
      if (Instruction *UserInst = dyn_cast<Instruction>(U)) {
        UserList.push_back(UserInst);
      }
    }

    for (Instruction *U : UserList) {
        vectorizeInstruction(U, I, nextReplacement);
    }

    // Splice returned block
    BasicBlock *ToBB = I->getParent();
    auto InsertPos = std::next(I->getIterator());
    ToBB->splice(
        InsertPos,
        tempBB,
        tempBB->begin(),
        tempBB->end()
    );
    I->eraseFromParent();
    tempBB->eraseFromParent();
  }

  // assumes the opcode is vectorizable, and that the operands are either vectors or vectorizable
  // returns new instruction
  llvm::BasicBlock *vectorizedReplace(llvm::Instruction* I, llvm::Value* oldVal, llvm::Value* newVal, Instruction **NextInst) {
    llvm::BasicBlock *tempBB = llvm::BasicBlock::Create(F->getContext(), "temp", F);
    llvm::IRBuilder<> Builder(tempBB);

    std::vector<Value *> newOperands;
    for (unsigned i = 0; i < I->getNumOperands(); ++i) {
      if (I->getOperand(i) == oldVal) {
        newOperands.push_back(newVal);
      } else if (!I->getOperand(i)->getType()->isVectorTy()) {
        newOperands.push_back(promoteScalarToVector(Builder, I->getOperand(i), GangSize));
      }
    }

    llvm::Instruction *newInst = nullptr;
    if (Instruction::isBinaryOp(I->getOpcode())) {
      newInst = BinaryOperator::Create((llvm::Instruction::BinaryOps)I->getOpcode(), newOperands[0], newOperands[1]);
    } else if (Instruction::isUnaryOp(I->getOpcode())) {
      newInst = UnaryOperator::Create((llvm::Instruction::UnaryOps)I->getOpcode(), newOperands[0]);
    } else if (Instruction::isCast(I->getOpcode())) {
      newInst = CastInst::Create((llvm::Instruction::CastOps)I->getOpcode(), newOperands[0], I->getType());
    }
    assert(newInst != nullptr);

    Builder.Insert(newInst);
    *NextInst = newInst;
    return tempBB;
  }

  llvm::BasicBlock* unvectorizedReplace(llvm::Instruction* I, llvm::Value* oldVal, llvm::Value* newVal, Instruction **NextInst) {
    llvm::BasicBlock *tempBB = llvm::BasicBlock::Create(F->getContext(), "temp", F);
    llvm::IRBuilder<> Builder(tempBB);

    unsigned changedOpIdx = 0;
    for (unsigned i = 0; i < I->getNumOperands(); ++i) {
      if (I->getOperand(i) == oldVal) {
        changedOpIdx = i;
        break;
      }
    }

    std::vector<Value *> newInsts;
    if (newVal->getType()->isVectorTy()) {
      VectorType *VecTy = cast<VectorType>(newVal->getType());
      for (unsigned i = 0; i < VecTy->getElementCount().getKnownMinValue(); ++i) {
        Value *extractedVal = Builder.CreateExtractElement(newVal, i);
        Instruction *newInst = I->clone();
        newInst->setOperand(changedOpIdx, extractedVal);
        newInsts.push_back(Builder.Insert(newInst));
      }
    } else {
      Value *Zero = Builder.getInt32(0);
      Type *newValType = newVal->getType();
      if (auto *castNewVal = llvm::dyn_cast<llvm::AllocaInst>(newVal)) {
        newValType = llvm::cast<llvm::ArrayType>(castNewVal->getAllocatedType());
      }
      for (unsigned i = 0; i < newValType->getArrayNumElements(); ++i) {
        Value *Idx = Builder.getInt32(i);
        Value *ElemPtr = Builder.CreateGEP(newValType, newVal, {Zero, Idx});
        Value *extractedVal = Builder.CreateLoad(newValType->getArrayElementType(), ElemPtr);
        Instruction *newInst = I->clone();
        newInst->setOperand(changedOpIdx, extractedVal);
        Builder.Insert(newInst);
        newInsts.push_back(newInst);
      }
    }

    if (newInsts[0]->getType()->isVoidTy()) {
      return tempBB;
    }
    if (newInsts[0]->getType()->isFloatingPointTy() || newInsts[0]->getType()->isIntegerTy()) {
      Type *EltTy = newInsts[0]->getType();
      VectorType *VecTy = VectorType::get(EltTy, newInsts.size(), false);
      Value *Vec = UndefValue::get(VecTy);
      for (unsigned i = 0; i < newInsts.size(); ++i) {
        Vec = Builder.CreateInsertElement(Vec, newInsts[i], i);
      }
      *NextInst = cast<llvm::Instruction>(Vec);
      return tempBB;
    }

    ArrayType *ArrTy = ArrayType::get(newInsts[0]->getType(), newInsts.size());
    Value *Zero = Builder.getInt32(0);
    Instruction *Alloc = Builder.CreateAlloca(ArrTy);
    for (unsigned i = 0; i < newInsts.size(); ++i) {
      Value *Idx = Builder.getInt32(i);
      Value *ElemPtr = Builder.CreateGEP(ArrTy, Alloc, {Zero, Idx}, "elem.ptr");
      Builder.CreateStore(newInsts[i], ElemPtr);
    }
    *NextInst = Alloc;
    return tempBB;
  }

  bool isVectorizableInstruction(Instruction *I) {
    switch (I->getOpcode()) {
      case Instruction::Add:
      case Instruction::FAdd:
      case Instruction::Sub:
      case Instruction::FSub:
      case Instruction::Mul:
      case Instruction::FMul:
      case Instruction::UDiv:
      case Instruction::SDiv:
      case Instruction::FDiv:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
      case Instruction::Shl:
      case Instruction::AShr:
      case Instruction::LShr:
        break;
      default:
        return false;
    }

    for (Value *Op : I->operands()) {
      Type *OpType = Op->getType();
      if (!OpType->isIntegerTy() && !OpType->isFloatingPointTy() && !OpType->isVectorTy()) {
        return false;
      }
    }
    return true;
  }

  Value* promoteScalarToVector(llvm::IRBuilder<> &Builder, Value *scalar, unsigned vecWidth) {
      Type *ScalarTy = scalar->getType();
      VectorType *VecTy = VectorType::get(ScalarTy, vecWidth, false);
      Value *UndefVec = UndefValue::get(VecTy);
      Value *Inserted = Builder.CreateInsertElement(UndefVec, scalar, Builder.getInt32(0));
      Value *SplatMask = ConstantVector::getSplat(
          ElementCount::getFixed(vecWidth), Builder.getInt32(0)
      );
      Value *SplatVec = Builder.CreateShuffleVector(Inserted, UndefVec, SplatMask);

      return SplatVec;
  }
};

llvm::PreservedAnalyses AddIndirection::run(llvm::Function &F,
                                       llvm::FunctionAnalysisManager &AM) {
  
  if (!isKernelToProcess(F))
    return llvm::PreservedAnalyses::all();

  // always run (Fix this later)
  // WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  // if (WIH != WorkitemHandlerType::LOOPS)
  //   return llvm::PreservedAnalyses::all();

  auto &TTI = AM.getResult<TargetIRAnalysis>(F);

  auto runner = AddIndirectionImpl(TTI);
  runner.runOnFunction(F);

  return PreservedAnalyses::none();
}


REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl

