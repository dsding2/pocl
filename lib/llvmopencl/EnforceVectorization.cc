// LLVM function pass to create loops that run all the work items
// in a work group while respecting barrier synchronization points.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen / Tampere University
//               2022-2025 Pekka Jääskeläinen / Intel Finland Oy
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

#include "CompilerWarnings.h"
#include <deque>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>
#include <unordered_map>
#include <unordered_set>
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Local.h>

#include "EnforceVectorization.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "WorkitemHandlerChooser.h"

POP_COMPILER_DIAGS

#include <array>
#include <map>
#include <vector>

#define DEBUG_TYPE "WIL"

#define PASS_NAME "enforce-vectorization"
#define PASS_CLASS pocl::EnforceVectorization
#define PASS_DESC                                                              \
  "Forces vectorization through a simple walk over the data flow graph"

// #define DEBUG_WORK_ITEM_LOOPS
// #define POCL_KERNEL_COMPILER_DUMP_CFGS

#define HAS_VALUE_OUTPUT(x) (x.valueOutput != nullptr)
#define HAS_ARRAY_OUTPUT(x) (!x.arrayOutput.empty())

namespace pocl {

using namespace llvm;

class EnforceVectorizationImpl : public pocl::WorkitemHandler {
public:
  EnforceVectorizationImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                           llvm::PostDominatorTree &PDT,
                           VariableUniformityAnalysisResult &VUA)
      : WorkitemHandler(), DT(DT), LI(LI), PDT(PDT), VUA(VUA) {}
  virtual bool runOnFunction(llvm::Function &F);

  // protected:
  // llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr) override;
  // llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr,
  //                                       size_t Dim) override;

private:
  struct GraphNode {
    llvm::Instruction *valueOutput = nullptr;
    std::vector<llvm::Instruction *> arrayOutput;
    std::unordered_set<llvm::Instruction *> children;

    // Used for Kahn's algorithm to generate a topological sort
    int in_degree = 0;

    // Determines whether loads should be converted to wide loads or gathers
    bool isContiguous = true;
  };

  using BasicBlockVector = std::vector<llvm::BasicBlock *>;
  using InstructionIndex = std::set<llvm::Instruction *>;
  using InstructionVec = std::vector<llvm::Instruction *>;
  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;
  VariableUniformityAnalysisResult &VUA;
  llvm::Module *M;
  llvm::Function *F;

  llvm::GlobalVariable *VectorizedGlobalIdVar;
  llvm::GlobalVariable *VectorizedLocalIdVar;
  int GangSize;
  int VectorizationDim;
  llvm::Type *VectorST;
  llvm::Constant *SequentialVec;
  llvm::Constant *StepVec;
  // first is the load inst, second is the (vectorized) replacement value
  std::vector<std::pair<llvm::Instruction *, llvm::Value *>> LoadInsts;

  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;
  std::array<llvm::GlobalVariable *, 3> LocalIdIterators;
  std::unordered_set<llvm::Instruction *> markedForDeletion;

  // contains all instructions that reference
  // get_global/local_id(vectorization_dim) pair.first is the instruction,
  // pair.second is a replacement value that loads from the vectorized ptr.
  std::vector<std::pair<llvm::Value *, llvm::Value *>> LocalIdInsts;
  std::vector<std::pair<llvm::Value *, llvm::Value *>> GlobalIdInsts;

  std::map<llvm::Instruction *, GraphNode> InstGraph;
  // Given a basic block, get a vector containing pairs of the form
  // <predecessor, mask value from predecessor>
  std::map<llvm::BasicBlock *, std::map<llvm::BasicBlock *, llvm::Value *>>
      BlockMaskPredecessors;
  std::map<llvm::BasicBlock *, llvm::Value *> BlockMasks;
  // associates each loop with the mask before the loop began (since the mask
  // needs to reset to that value afterwards)
  std::map<llvm::Loop *, llvm::Value *> LoopMasks;
  // Store all created calls that need a mask to be filled in later.
  std::set<std::pair<llvm::CallInst *, int>> maskedCalls;
  std::unordered_set<llvm::BranchInst *> branchInsts;

  // set of blocks with loop exiting branches
  std::set<llvm::BasicBlock *> loopBranches;

  bool processFunction(llvm::Function &F);

  void markLoop(Loop *L);
  void handleLoopMask(Loop *L);
  Constant *createConstantMask(IRBuilder<> &Builder, int N);

  Value *createVectorScalarAdd(IRBuilder<> &builder, Value *Vec, Value *Scalar);
  void vectorizedHandleWorkitemFunctions();

  Instruction *ConvertOpToMaskedIntrinsic(IRBuilder<> &Builder, Instruction *I,
                                          Value *op1, Value *op2);
  bool checkContiguous(llvm::Instruction *I, llvm::Instruction *oldVal);
  void vectorizeInstruction(llvm::Instruction *I);
  void branchReplace(llvm::BranchInst *I);
  void vectorizedReplace(llvm::Instruction *I);
  void unvectorizedReplace(llvm::Instruction *I);
  bool isVectorizableInstruction(Instruction *I);
  void
  traverseInstructionTree(Instruction *I,
                          std::unordered_set<llvm::Instruction *> &visited);

  void fixMultiRegionVariables(ParallelRegion *Region);
  void addContextSaveRestore(llvm::Instruction *instruction);
  void releaseParallelRegions();

  Constant *makeSequentialVector();

  void transformIdStores(BasicBlock *BB);
  void transformForInc(BasicBlock *BB);
  void transformIdLoads();
  Instruction *findIncrementOfGlobal(BasicBlock *BB, GlobalVariable *GV);
  void findLoadsOfGlobal(GlobalVariable *GV, std::vector<Instruction *> &Loads);
  void transformControlFlow(BasicBlock *BB);

  bool blockHasTag(const BasicBlock *BB, StringRef Role);

  // Given a typical predicate, it constructs the prefix from the predicate
  // mask. That is, a predicate of 1101 becomes 1100, to represent the fact that
  // the loop should exit after the first 0.
  Value *constructLoopPredicatePrefix(IRBuilder<> &Builder, Value *predicate);

  bool privatizeContext();
};

bool EnforceVectorizationImpl::runOnFunction(Function &Func) {
  M = Func.getParent();
  F = &Func;
  Initialize(cast<Kernel>(&Func));

  // TODO add proper inference of gang size
  GangSize = 2;
  VectorizationDim = 0;
  // if (WGLocalSizeX >= WGLocalSizeY && WGLocalSizeX >= WGLocalSizeZ) {
  //   VectorizationDim = 0;
  // } else if (WGLocalSizeY >= WGLocalSizeX && WGLocalSizeY >= WGLocalSizeZ) {
  //   VectorizationDim = 1;
  // } else {
  //   VectorizationDim = 2;
  // }
  VectorST = VectorType::get(ST, GangSize, false);

  SequentialVec = makeSequentialVector();
  StepVec = ConstantVector::getSplat(ElementCount::getFixed(GangSize),
                                     ConstantInt::get(ST, GangSize));

  GlobalIdIterators = {M->getGlobalVariable(GID_G_NAME(0), ST),
                       M->getGlobalVariable(GID_G_NAME(1), ST),
                       M->getGlobalVariable(GID_G_NAME(2), ST)};

  LocalIdIterators = {M->getGlobalVariable(LID_G_NAME(0), ST),
                      M->getGlobalVariable(LID_G_NAME(1), ST),
                      M->getGlobalVariable(LID_G_NAME(2), ST)};

  VectorizedGlobalIdVar = cast<GlobalVariable>(
      M->getOrInsertGlobal(GID_G_NAME_VECTORIZED, VectorST));
  VectorizedLocalIdVar = cast<GlobalVariable>(
      M->getOrInsertGlobal(LID_G_NAME_VECTORIZED, VectorST));

  for (Loop *L : LI.getLoopsInPreorder()) {
    markLoop(L);
  }

  bool Changed = processFunction(Func);

  Changed |= handleLocalMemAllocas();

  Changed |= fixUndominatedVariableUses(DT, Func);

  Changed |= privatizeContext();
  return Changed;
}

Constant *EnforceVectorizationImpl::createConstantMask(IRBuilder<> &Builder,
                                                       int N) {
  Type *I1Ty = Builder.getInt1Ty();
  VectorType *MaskTy = FixedVectorType::get(I1Ty, GangSize);

  SmallVector<Constant *, 16> Elems;

  unsigned Active = std::min(N, GangSize);

  for (unsigned i = 0; i < GangSize; ++i) {
    bool Bit = (i < Active);
    Elems.push_back(ConstantInt::get(I1Ty, Bit));
  }

  return ConstantVector::get(Elems);
}

void EnforceVectorizationImpl::markLoop(Loop *L) {
  SmallVector<BasicBlock *, 4> exitingBlocks;
  L->getExitingBlocks(exitingBlocks);

  assert(exitingBlocks.size() == 1);
  BranchInst *br = dyn_cast<BranchInst>(exitingBlocks[0]->getTerminator());
  assert(br && br->isConditional());
  Value *cond = br->getCondition();

  bool trueContinuesLoop = L->contains(br->getSuccessor(0));
  if (!trueContinuesLoop) {
    IRBuilder<> Builder(br->getParent());
    Builder.SetInsertPoint(br);
    Value *condInv = Builder.CreateNot(cond);

    BasicBlock *tmp = br->getSuccessor(0);
    br->setSuccessor(0, br->getSuccessor(1));
    br->setSuccessor(1, tmp);
    br->setCondition(condInv);
  }
  loopBranches.insert(br->getParent());
}

Constant *EnforceVectorizationImpl::makeSequentialVector() {
  std::vector<Constant *> Elements;
  Elements.reserve(GangSize);
  for (unsigned i = 0; i < GangSize; i++) {
    Elements.push_back(ConstantInt::get(ST, i));
  }
  return ConstantVector::get(Elements);
}

void EnforceVectorizationImpl::traverseInstructionTree(
    Instruction *I, std::unordered_set<llvm::Instruction *> &visited) {
  visited.insert(I);
  for (User *U : I->users()) {
    Instruction *ChildInst = dyn_cast<Instruction>(U);
    if (ChildInst) {
      InstGraph[ChildInst].in_degree += 1;
      InstGraph[I].children.insert(ChildInst);
      if (visited.count(ChildInst) == 0) {
        traverseInstructionTree(ChildInst, visited);
      }
    }
  }
}

bool EnforceVectorizationImpl::checkContiguous(Instruction *I,
                                               Instruction *oldVal) {
  auto IOpcode = I->getOpcode();
  if (IOpcode == Instruction::Add || IOpcode == Instruction::Sub ||
      IOpcode == Instruction::GetElementPtr) {
    return InstGraph[oldVal].isContiguous;
  }

  Instruction *Grandparent;
  const APInt *Shift;
  if (PatternMatch::match(
          I, PatternMatch::m_AShr(
                 PatternMatch::m_Shl(PatternMatch::m_Instruction(Grandparent),
                                     PatternMatch::m_APInt(Shift)),
                 PatternMatch::m_APInt(Shift)))) {
    return InstGraph[Grandparent].isContiguous;
  }
  return false;
}

void EnforceVectorizationImpl::vectorizeInstruction(llvm::Instruction *I) {
  // InstGraph[I].mask = InstGraph[oldVal].mask;
  // InstGraph[I].isContiguous = checkContiguous(I, oldVal);

  // Handle branches
  if (BranchInst *br = dyn_cast<BranchInst>(I)) {
    branchReplace(br);
  } else if (isVectorizableInstruction(I)) {
    vectorizedReplace(I);
  } else {
    unvectorizedReplace(I);
  }
}

bool EnforceVectorizationImpl::blockHasTag(const BasicBlock *BB, StringRef Role) {
  auto *M = BB->getTerminator()->getMetadata("myrole");
  auto *S = M ? dyn_cast<MDString>(M->getOperand(0)) : nullptr;
  return S && S->getString() == Role;
}

void EnforceVectorizationImpl::branchReplace(llvm::BranchInst *br) {
  if (!br->isConditional()) {
    return;
  }

  BasicBlock *BB = br->getParent();

  Value *cond = br->getCondition();
  Instruction *condInst = dyn_cast<Instruction>(cond);
  if (condInst && InstGraph.count(condInst) > 0) {
    // conversion from array output to value output
    if (!HAS_VALUE_OUTPUT(InstGraph[condInst])) {
      std::vector<Instruction *> &oldValOutputs =
          InstGraph[condInst].arrayOutput;
      IRBuilder<> Builder(oldValOutputs.back()->getNextNode());
      Type *elemTy = oldValOutputs[0]->getType();
      VectorType *vecTy = VectorType::get(elemTy, GangSize, false);
      Value *vec = UndefValue::get(vecTy);
      for (unsigned i = 0; i < GangSize; i++) {
        vec = Builder.CreateInsertElement(vec, oldValOutputs[i],
                                          Builder.getInt32(i));
      }
      InstGraph[condInst].valueOutput = cast<Instruction>(vec);
    }

    condInst = InstGraph[condInst].valueOutput;
    IRBuilder<> Builder(condInst->getParent());

    if (loopBranches.count(br->getParent()) > 0) {
      Builder.SetInsertPoint(br);
      // For loop predicates, any false means that all future mask values must also be false.
      // TODO: change this to only apply to branches in the for_cond blocks.
      // For other blocks, you must set it up such that if a mask value is false, it will be false forever.
      if (blockHasTag(br->getParent(), "pregion_for_cond")) {
        Value *newLoopPredicate = constructLoopPredicatePrefix(Builder, condInst);
        cast<PHINode>(BlockMasks[br->getSuccessor(0)])->addIncoming(newLoopPredicate, BB);
        // BlockMaskPredecessors[br->getSuccessor(1)][br->getParent()] = nullptr;
        Value *newBrCond = Builder.CreateOrReduce(newLoopPredicate);
        // BranchInst *newBr = Builder.CreateCondBr(newBrCond, br->getSuccessor(0), br->getSuccessor(1));

        Value *maskAllTrue = Builder.CreateAndReduce(BlockMasks[BB]);
        Value *finalCond = Builder.CreateAnd(newBrCond, maskAllTrue);
        br->setCondition(finalCond);
        // markedForDeletion.insert(br);
      } else {
        Value *newMask = Builder.CreateAnd(BlockMasks[BB], condInst);
        cast<PHINode>(BlockMasks[br->getSuccessor(0)])->addIncoming(newMask, BB);
        Value *newBrCond = Builder.CreateOrReduce(newMask);
        br->setCondition(newBrCond);
      }


    } else {
      Builder.SetInsertPoint(condInst->getNextNode());
      Value *inv = Builder.CreateNot(condInst);
      cast<PHINode>(BlockMasks[br->getSuccessor(0)])->addIncoming(condInst, BB);
      cast<PHINode>(BlockMasks[br->getSuccessor(1)])->addIncoming(inv, BB);
      branchInsts.insert(br);
    }
  }
}

#define SPLAT_OPERAND(idx)                                                     \
  if (!newOperands[idx]->getType()->isVectorTy()) {                            \
    newOperands[idx] = Builder.CreateVectorSplat(GangSize, newOperands[idx]);  \
  }
// assumes the opcode is vectorizable, and that the operands are either vectors
// or vectorizable returns new instruction
void EnforceVectorizationImpl::vectorizedReplace(llvm::Instruction *I) {

  std::vector<Value *> newOperands;

  // We need to insert all new instructions after the transformations of all of
  // their operands. Thus, find the latest operand.
  Instruction *insertPoint = I->getNextNode();
  // for (unsigned i = 0; i < I->getNumOperands(); ++i) {
  //   auto *operandInst = dyn_cast<Instruction>(I->getOperand(i));
  //   if (!operandInst || !InstGraph.count(operandInst))
  //     continue;

  //   Instruction *candidate =
  //       !HAS_VALUE_OUTPUT(InstGraph[operandInst])
  //           ? candidate = InstGraph[operandInst].arrayOutput.back()
  //           : candidate = InstGraph[operandInst].valueOutput;

  //   if (candidate && latestOperand->getParent() == candidate->getParent() &&
  //       latestOperand->comesBefore(candidate)) {
  //     latestOperand = candidate;
  //   }
  // }
  llvm::IRBuilder<> Builder(insertPoint);

  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    Instruction *operandInst = dyn_cast<Instruction>(I->getOperand(i));
    if (operandInst && InstGraph.count(operandInst) > 0) {
      // convert arrayOutput to valueOutput
      if (!HAS_VALUE_OUTPUT(InstGraph[operandInst])) {
        std::vector<Instruction *> &oldValOutputs =
            InstGraph[operandInst].arrayOutput;
        Type *elemTy = oldValOutputs[0]->getType();
        VectorType *vecTy = VectorType::get(elemTy, GangSize, false);
        Value *vec = UndefValue::get(vecTy);
        Builder.SetInsertPoint(oldValOutputs.back()->getNextNode());
        for (unsigned i = 0; i < GangSize; i++) {
          vec = Builder.CreateInsertElement(vec, oldValOutputs[i],
                                            Builder.getInt32(i));
        }
        Builder.SetInsertPoint(insertPoint);
        InstGraph[operandInst].valueOutput = cast<Instruction>(vec);
      }
      newOperands.push_back(InstGraph[operandInst].valueOutput);
    } else {
      newOperands.push_back(I->getOperand(i));
    }
  }

  llvm::Instruction *newInst = nullptr;
  if (I->getOpcode() == Instruction::GetElementPtr) {
    // Again, I can't figure out the correct semantics for GEP, so it's disabled
    // for now
    assert(false);
    // VectorType *vectorizedType = VectorType::get(I->getType(), GangSize,
    // false); SPLAT_OPERAND(0); SPLAT_OPERAND(1); newInst =
    // cast<Instruction>(Builder.CreateGEP(I->getType(), newOperands[0],
    // newOperands[1]));
  } else if (I->getOpcode() == Instruction::Load) {
    VectorType *vectorizedType = VectorType::get(I->getType(), GangSize, false);
    SPLAT_OPERAND(0);
    auto newCallInst =
        Builder.CreateMaskedGather(vectorizedType, newOperands[0], Align(4),
                                   NULL, PoisonValue::get(vectorizedType));
    maskedCalls.insert(std::make_pair(newCallInst, 2));
    newInst = cast<Instruction>(newCallInst);
  } else if (I->getOpcode() == Instruction::Store) {
    SPLAT_OPERAND(0);
    SPLAT_OPERAND(1);
    auto newCallInst = Builder.CreateMaskedScatter(
        newOperands[0], newOperands[1], Align(4), NULL);
    maskedCalls.insert(std::make_pair(newCallInst, 3));
    newInst = cast<Instruction>(newCallInst);
  } else if (isa<CmpInst>(I)) {
    SPLAT_OPERAND(0);
    SPLAT_OPERAND(1);
    CmpInst *ICMP = cast<CmpInst>(I);
    newInst = cast<Instruction>(Builder.CreateICmp(
        ICMP->getPredicate(), newOperands[0], newOperands[1]));
  } else if (Instruction::isBinaryOp(I->getOpcode())) {
    SPLAT_OPERAND(0);
    SPLAT_OPERAND(1);
    newInst = cast<Instruction>(
        Builder.CreateBinOp((llvm::Instruction::BinaryOps)I->getOpcode(),
                            newOperands[0], newOperands[1]));
  } else if (Instruction::isUnaryOp(I->getOpcode())) {
    SPLAT_OPERAND(0);
    newInst = cast<Instruction>(Builder.CreateUnOp(
        (llvm::Instruction::UnaryOps)I->getOpcode(), newOperands[0]));
  } else if (Instruction::isCast(I->getOpcode())) {
    SPLAT_OPERAND(0);
    newInst = cast<Instruction>(
        Builder.CreateCast((llvm::Instruction::CastOps)I->getOpcode(),
                           newOperands[0], I->getType()));
  } else {
    assert(false);
  }

  markedForDeletion.insert(I);
  InstGraph[I].valueOutput = newInst;
}

void EnforceVectorizationImpl::unvectorizedReplace(llvm::Instruction *I) {
  // We need to insert all new instructions after the transformations of all of
  // their operands. Thus, find the latest operand.
  Instruction *insertPoint = I->getNextNode();
  // for (unsigned i = 0; i < I->getNumOperands(); ++i) {
  //   auto *operandInst = dyn_cast<Instruction>(I->getOperand(i));
  //   if (!operandInst || InstGraph.count(operandInst) == 0)
  //     continue;

  //   Instruction *Candidate = HAS_ARRAY_OUTPUT(InstGraph[operandInst])
  //                                ? InstGraph[operandInst].arrayOutput.back()
  //                                : InstGraph[operandInst].valueOutput;

  //   if (Candidate && latestOperand->getParent() == Candidate->getParent() &&
  //       latestOperand->comesBefore(Candidate))
  //     latestOperand = Candidate;
  // }
  IRBuilder<> Builder(insertPoint);

  std::unordered_map<int, std::vector<Instruction *>> changedOperands;

  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    Instruction *operandInst = dyn_cast<Instruction>(I->getOperand(i));
    if (operandInst && InstGraph.count(operandInst) > 0) {
      // convert valueOutput to vectorOutput
      std::vector<Instruction *> unpackedOperand;
      if (!HAS_ARRAY_OUTPUT(InstGraph[operandInst])) {
        Builder.SetInsertPoint(
            InstGraph[operandInst].valueOutput->getNextNode());
        for (unsigned i = 0; i < GangSize; ++i) {
          Instruction *ExtractedElem =
              cast<Instruction>(Builder.CreateExtractElement(
                  InstGraph[operandInst].valueOutput, i));
          unpackedOperand.push_back(ExtractedElem);
        }
        InstGraph[operandInst].arrayOutput = unpackedOperand;
        Builder.SetInsertPoint(insertPoint);
      }

      changedOperands[i] = InstGraph[operandInst].arrayOutput;
    }
  }

  for (int i = 0; i < GangSize; ++i) {
    Instruction *newInst = I->clone();
    for (auto &[j, changedOperandArr] : changedOperands) {
      newInst->setOperand(j, changedOperandArr[i]);
    }
    InstGraph[I].arrayOutput.push_back(newInst);
    Builder.Insert(newInst);
  }

  markedForDeletion.insert(I);
}

bool EnforceVectorizationImpl::isVectorizableInstruction(Instruction *I) {
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
  case Instruction::ICmp:
  case Instruction::Load:
  case Instruction::Store:
    // In theory, there should be a vectorized version of GEP, but I can't
    // figure out the semantics case Instruction::GetElementPtr:
    break;
  default:
    return false;
  }

  for (Value *Op : I->operands()) {
    Type *OpType = Op->getType();
    if (!OpType->isIntegerTy() && !OpType->isFloatingPointTy() &&
        !OpType->isVectorTy() && !OpType->isPointerTy()) {
      return false;
    }
  }
  return true;
}

Value *EnforceVectorizationImpl::createVectorScalarAdd(IRBuilder<> &builder,
                                                       Value *Vec,
                                                       Value *Scalar) {
  auto *VecTy = cast<VectorType>(Vec->getType());
  unsigned NumElems = VecTy->getElementCount().getFixedValue();

  Value *Splat = builder.CreateVectorSplat(NumElems, Scalar);
  return builder.CreateAdd(Vec, Splat);
}

// Set up the vectorized global and local id. That is, initialize the vectors
// <0, 1, 2,.., GangSize-1>.
void EnforceVectorizationImpl::transformIdStores(BasicBlock *BB) {
  // find init of global variable (which contains the offset as an operand)
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BB->getFirstInsertionPt());

  std::vector<StoreInst *> Results;
  for (Instruction &I : *BB) {
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->getPointerOperand() == GlobalIdGlobals[VectorizationDim]) {
        Builder.SetInsertPoint(SI);
        Builder.CreateStore(createVectorScalarAdd(Builder, SequentialVec,
                                                  SI->getValueOperand()),
                            VectorizedGlobalIdVar);
        break;
      } else if (SI->getPointerOperand() == LocalIdGlobals[VectorizationDim]) {
        Builder.SetInsertPoint(SI);
        Builder.CreateStore(createVectorScalarAdd(Builder, SequentialVec,
                                                  SI->getValueOperand()),
                            VectorizedLocalIdVar);
      }
    }
  }

  // TODO fix this (why was this like this??)
  // Builder.CreateStore(SequentialVec, VectorizedLocalIdVar);
}

Instruction *
EnforceVectorizationImpl::findIncrementOfGlobal(BasicBlock *BB,
                                                GlobalVariable *GV) {
  for (Instruction &I : *BB) {
    // Match: %x = load i64, ptr @GV
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->getPointerOperand()->stripPointerCasts() != GV)
        continue;

      // Next instruction must be %y = add i64 %x, 1
      if (auto *AI = dyn_cast<BinaryOperator>(LI->getNextNode())) {
        Value *X;
        if (!PatternMatch::match(
                AI, PatternMatch::m_Add(PatternMatch::m_Value(X),
                                        PatternMatch::m_ConstantInt<1>())))
          continue;

        if (X != LI) // ensure add uses the load
          continue;
        // Next instruction must be store %y, ptr @GV
        if (auto *SI = dyn_cast<StoreInst>(AI->getNextNode())) {
          if (SI->getValueOperand() == AI &&
              SI->getPointerOperand()->stripPointerCasts() == GV) {
            return AI;
          }
        }
      }
    }
  }
  return nullptr;
}

// Increment the iterator vector by GangSize
void EnforceVectorizationImpl::transformForInc(BasicBlock *BB) {
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BB->getFirstInsertionPt());
  llvm::GlobalVariable *LocalIdVar = LocalIdIterators[VectorizationDim];
  llvm::GlobalVariable *GlobalIdVar = GlobalIdIterators[VectorizationDim];

  Instruction *LocalIteratorInc = findIncrementOfGlobal(BB, LocalIdVar);
  if (LocalIteratorInc) {
    for (int i = 0; i < 2; i++) {
      if (ConstantInt *CI =
              dyn_cast<ConstantInt>(LocalIteratorInc->getOperand(i))) {
        LocalIteratorInc->setOperand(i, ConstantInt::get(ST, GangSize));
        break;
      }
    }
    Builder.CreateStore(
        Builder.CreateAdd(Builder.CreateLoad(VectorST, VectorizedLocalIdVar),
                          StepVec),
        VectorizedLocalIdVar);
  }

  Value *newGlobalValue;
  Instruction *GlobalIteratorInc = findIncrementOfGlobal(BB, GlobalIdVar);
  if (GlobalIteratorInc) {
    for (int i = 0; i < 2; i++) {
      if (ConstantInt *CI =
              dyn_cast<ConstantInt>(GlobalIteratorInc->getOperand(i))) {
        GlobalIteratorInc->setOperand(i, ConstantInt::get(ST, GangSize));
        break;
      }
    }
    Builder.CreateStore(
        Builder.CreateAdd(Builder.CreateLoad(VectorST, VectorizedGlobalIdVar),
                          StepVec),
        VectorizedGlobalIdVar);
  }

  // Generate new mask
  if (LocalIteratorInc) {
    // Value *sequential = makeSequentialVector();
    // Value *trailing = Builder.CreateSub(Builder.getInt32(WGLocalSizeX),
    // LocalIteratorInc); Value *trailingSplat =
    // Builder.CreateVectorSplat(GangSize, trailing); Value *mask =
    // Builder.CreateICmpULT(sequential, trailingSplat);
  }
}

void EnforceVectorizationImpl::findLoadsOfGlobal(
    GlobalVariable *GV, std::vector<Instruction *> &Loads) {
  for (User *U : GV->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U)) {
      BasicBlock *BB = LI->getParent();
      if (auto *M = BB->getTerminator()->getMetadata("myrole")) {
        auto *S = dyn_cast<MDString>(M->getOperand(0));
        if (S && (S->getString() == "pregion_for_inc" ||
                  S->getString() == "pregion_for_init")) {
          continue;
        }
      }
      Loads.push_back(LI);
    }
  }
}

Value *
EnforceVectorizationImpl::constructLoopPredicatePrefix(IRBuilder<> &Builder,
                                                       Value *predicate) {
  auto *VTy = VectorType::get(Builder.getInt1Ty(), GangSize, false);

  // Compute prefix-and of A.
  SmallVector<Value *> Prefix(GangSize);
  Prefix[0] = Builder.CreateExtractElement(predicate, Builder.getInt32(0));

  for (unsigned I = 1; I < GangSize; ++I) {
    Value *Cur = Builder.CreateExtractElement(predicate, Builder.getInt32(I));
    Prefix[I] = Builder.CreateAnd(Prefix[I - 1], Cur);
  }

  Value *PrefixMask = PoisonValue::get(VTy);

  for (unsigned I = 0; I < GangSize; ++I) {
    PrefixMask =
        Builder.CreateInsertElement(PrefixMask, Prefix[I], Builder.getInt32(I));
  }

  return PrefixMask;
}

// Starting with every load of the global and local id values, recursively walk
// the data flow graph, applying vectorization to every instruction found.
void EnforceVectorizationImpl::transformIdLoads() {
  IRBuilder<> Builder(M->getContext());

  // initialize masks with blank PHI nodes
  for (BasicBlock &BB : *F) {
    Builder.SetInsertPoint(BB.getFirstInsertionPt());
    unsigned numPreds = std::distance(pred_begin(&BB), pred_end(&BB));
    if (numPreds == 0) {
      BlockMasks[&BB] = createConstantMask(Builder, GangSize);
    } else {
      BlockMasks[&BB] = Builder.CreatePHI(
          VectorType::get(Builder.getInt1Ty(), GangSize, false), numPreds);
    }
  }

  std::unordered_set<Instruction *> AllVectorizableLoads;
  std::vector<Instruction *> GlobalIteratorLoads;
  std::vector<Instruction *> LocalIteratorLoads;
  std::unordered_set<Instruction *> visited;
  findLoadsOfGlobal(LocalIdIterators[VectorizationDim], LocalIteratorLoads);
  for (Instruction *LoadInst : LocalIteratorLoads) {
    Builder.SetInsertPoint(LoadInst);
    Instruction *globalLoad =
        Builder.CreateLoad(VectorST, VectorizedLocalIdVar);
    InstGraph[LoadInst].valueOutput = globalLoad;
    AllVectorizableLoads.insert(LoadInst);
    traverseInstructionTree(LoadInst, visited);
  }

  findLoadsOfGlobal(GlobalIdIterators[VectorizationDim], GlobalIteratorLoads);
  for (Instruction *LoadInst : GlobalIteratorLoads) {
    Builder.SetInsertPoint(LoadInst);
    Instruction *globalLoad =
        Builder.CreateLoad(VectorST, VectorizedGlobalIdVar);
    InstGraph[LoadInst].valueOutput = globalLoad;
    AllVectorizableLoads.insert(LoadInst);
    traverseInstructionTree(LoadInst, visited);
  }

  // Topological sort
  SmallVector<Instruction *> Worklist;
  for (auto &LoadInst : AllVectorizableLoads) {
    Worklist.push_back(LoadInst);
  }
  SmallVector<Instruction *> TopoOrder;
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    TopoOrder.push_back(I);
    for (Instruction *User : InstGraph[I].children) {
      if (--(InstGraph[User].in_degree) == 0) {
        Worklist.push_back(User);
      }
    }
  }

  // Masking is handled in two stages. The first stage happens during the
  // recursive vectorizeInstruction, which figures out the masks for conditional
  // branches, and sets up needed information for CFG loops.
  for (Instruction *I : TopoOrder) {
    // Don't try to vectorize the initial loads from the global/local IDs.
    if (AllVectorizableLoads.count(I) == 0) {
      vectorizeInstruction(I);
    }
  }

  llvm::SmallVector<llvm::Instruction *> deletion_vec(markedForDeletion.begin(),
                                                    markedForDeletion.end());
  for (int i = 0; i < deletion_vec.size(); i++) {
    Instruction *inst = deletion_vec[i];
    inst->replaceAllUsesWith(PoisonValue::get(inst->getType()));
    inst->eraseFromParent();
  }

  // This is the second stage, which traverses the blocks in approximate order,
  // and fills out the masking. Generally, just use a phi node set to inherit
  // the right mask from the predecessor. In the case of loops, the
  // preprocessing in the first stage should avoid dependency cycles.
  auto updatePHI = [](PHINode *Phi, BasicBlock *BB, Value *V) -> bool {
    int idx = Phi->getBasicBlockIndex(BB);
    if (!V) {
      V = PoisonValue::get(Phi->getType());
    }

    if (idx < 0) {
      Phi->addIncoming(V, BB);
      return V != PoisonValue::get(Phi->getType());
    }
    if (Phi->getIncomingValue(idx) == V)
      return false;

    Phi->setIncomingValue(idx, V);
    return true;
  };

  // Right now, we take all enabled as default. This breaks if GangSize <
  // GlobalSize, and maybe some other edge cases
  Builder.SetInsertPoint(F->getEntryBlock().begin());
  // BlockMasks[&F->getEntryBlock()] = createConstantMask(Builder, GangSize);

  std::deque<BasicBlock *> worklist;
  for (BasicBlock *BB : successors(&F->getEntryBlock())) {
    worklist.push_back(BB);
  }

  while (!worklist.empty()) {
    BasicBlock *BB = worklist.front();
    worklist.pop_front();
    bool changed = false;

    IRBuilder<> Builder(&*BB->begin());
    PHINode *phi = dyn_cast<PHINode>(BlockMasks[BB]);
    if (!phi) {
      continue;
    }

    // For each predecessor, if it is loop exiting, then override the
    // inherited mask value to be the mask of the loop header (i.e. reset the
    // mask to the start of the loop) Otherwise, simply inherit the last
    // block's mask if not already set.

    for (BasicBlock *Pred : predecessors(BB)) {
      Value *Incoming = nullptr;
      for (Loop *L = LI.getLoopFor(Pred); L; L = L->getParentLoop()) {
        if (L->isLoopExiting(Pred) && !L->contains(BB) && LoopMasks[L]) {
          Incoming = LoopMasks[L];
          break;
        }
      }
      if (Incoming) {
        changed |= updatePHI(phi, Pred, Incoming);
      }

      if (phi->getBasicBlockIndex(Pred) >= 0) {
        continue;
      }
      changed |= updatePHI(phi, Pred, BlockMasks[Pred]);
    }

    // Handle loop mask. At the exit of the loop, the mask should reset to the
    // mask value before the loop, requiring this additional work to maintain
    // that value
    for (Loop *L = LI.getLoopFor(BB); L; L = L->getParentLoop()) {
      // Assumes that each loop header only has one direct predecessor. If this
      // assumption breaks, change this to regenerate the phi clone if changed == true instead.
      if (LoopMasks.count(L) == 0 && L->getHeader() == BB) {
        PHINode *phiClone = cast<PHINode>(phi->clone());
        for (BasicBlock *Pred : predecessors(BB)) {
          if (L->contains(Pred)) {
            phiClone->setIncomingValueForBlock(Pred, phiClone);
          }
        }
        LoopMasks[L] = phiClone;
        Builder.Insert(phiClone);
        changed = true;
        for (BasicBlock *LoopBB : L->getBlocks()) {
          worklist.push_back(LoopBB);
        }
      }
    }
    // BlockMasks[BB] = phi;

    if (changed) {
      for (BasicBlock *succ : successors(BB)) {
        worklist.push_back(succ);
      }
    }
  }

  // After setting up all the masks with the phi, we need to merge branches into
  // a single straight line path.

  // TODO: This breaks down if you have nested control flow.
  for (BranchInst *br : branchInsts) {
    BasicBlock *trueBB = br->getSuccessor(0);
    BasicBlock *falseBB = br->getSuccessor(1);
    BasicBlock *mergeBB = PDT.findNearestCommonDominator(trueBB, falseBB);

    std::unordered_set<BasicBlock *> trueBBMergePreds;
    std::unordered_set<BasicBlock *> falseBBMergePreds;
    // find all immediate predecessors to the merge point that are on the path
    // from each branch
    for (BasicBlock *Pred : predecessors(mergeBB)) {
      if (Pred == mergeBB)
        continue;

      SmallPtrSet<BasicBlock *, 1> exclusionSet({mergeBB});
      if (isPotentiallyReachable(trueBB, Pred, &exclusionSet)) {
        trueBBMergePreds.insert(Pred);
      }
      if (isPotentiallyReachable(falseBB, Pred, &exclusionSet)) {
        falseBBMergePreds.insert(Pred);
      }
    }

    // For the false branch, we need to change the mask to the false predicate
    // mask.
    PHINode *falsePhiNode = cast<PHINode>(BlockMasks[falseBB]);
    Value *falseMask = falsePhiNode->getIncomingValueForBlock(br->getParent());
    falsePhiNode->removeIncomingValue(br->getParent(), false);

    // Then, we set up the true branch by rerouting the merge block predecessor
    // terminators to the beginning of the false branch. Then, we set the masks
    // for the false branch.
    // TODO: what if there are no blocks in the true branch?
    PHINode *MergeMask = cast<PHINode>(BlockMasks[mergeBB]);
    for (BasicBlock *trueBBMergePred : trueBBMergePreds) {
      BranchInst *mergePredBr =
          cast<BranchInst>(trueBBMergePred->getTerminator());
      for (int i = 0; i < mergePredBr->getNumSuccessors(); i++) {
        if (mergePredBr->getSuccessor(i) == mergeBB) {
          mergePredBr->setSuccessor(i, falseBB);
        }
      }
      MergeMask->removeIncomingValue(trueBBMergePred, false);
      falsePhiNode->addIncoming(falseMask, trueBBMergePred);
    }

    // Finally, for the mask for the merge block, we need to set them to the
    // correct values (that is, the mask before the predicate). If the false
    // branch has no blocks (i.e. points directly to the merge block), we set
    // the mask differently
    if (falseBBMergePreds.size() == 0) {
      for (BasicBlock *trueBBMergePred : trueBBMergePreds) {
        MergeMask->setIncomingValueForBlock(trueBBMergePred,
                                            BlockMasks[br->getParent()]);
      }
    } else {
      for (BasicBlock *falseBBMergePred : falseBBMergePreds) {
        MergeMask->setIncomingValueForBlock(falseBBMergePred,
                                            BlockMasks[br->getParent()]);
      }
    }

    Builder.SetInsertPoint(br);
    Builder.CreateBr(trueBB);
    br->eraseFromParent();
  }

  for (auto &[maskedCall, maskIdx] : maskedCalls) {
    maskedCall->setArgOperand(maskIdx, BlockMasks[maskedCall->getParent()]);
  }
}

void EnforceVectorizationImpl::transformControlFlow(BasicBlock *BB) {
  if (loopBranches.count(BB) > 0) {
    return;
  }
}

bool EnforceVectorizationImpl::privatizeContext() {
  CreateBuilder(Builder, F->getEntryBlock());
  if (VectorizedGlobalIdVar != nullptr) {
    AllocaInst *VectorizedGlobalIdAlloca =
        Builder.CreateAlloca(VectorST, 0, GID_G_NAME_VECTORIZED);
    for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
      for (BasicBlock::iterator ii = i->begin(), ee = i->end(); ii != ee;
           ++ii) {
        ii->replaceUsesOfWith(VectorizedGlobalIdVar, VectorizedGlobalIdAlloca);
      }
    }
  }

  if (VectorizedLocalIdVar != nullptr) {
    AllocaInst *VectorizedLocalIdAlloca =
        Builder.CreateAlloca(VectorST, 0, LID_G_NAME_VECTORIZED);
    for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
      for (BasicBlock::iterator ii = i->begin(), ee = i->end(); ii != ee;
           ++ii) {
        ii->replaceUsesOfWith(VectorizedLocalIdVar, VectorizedLocalIdAlloca);
      }
    }
  }
  M->eraseGlobalVariable(VectorizedGlobalIdVar);
  M->eraseGlobalVariable(VectorizedLocalIdVar);

  return true;
}

bool EnforceVectorizationImpl::processFunction(Function &F) {
  // vectorizedHandleWorkitemFunctions();
  std::vector<BasicBlock *> forInitBlocks;
  std::vector<BasicBlock *> forBodyBlocks;
  std::vector<BasicBlock *> forIncBlocks;
  for (auto &BB : F) {
    if (auto *M = BB.getTerminator()->getMetadata("myrole")) {
      auto *S = dyn_cast<MDString>(M->getOperand(0));
      if (!S) {
        continue;
      }
      if (S->getString() == "pregion_for_inc") {
        // forIncBlocks.push_back(&BB);
        transformForInc(&BB);
      } else {
        transformIdStores(&BB);
      }
      // if (S->getString() == "pregion_for_entry") {
      //   // Note to self: search for other linked blocks when dealing with
      //   this
      //   // block
      //   forBodyBlocks.push_back(&BB);
      // }
      // if (S->getString() == "pregion_for_init") {
      //   forInitBlocks.push_back(&BB);
      // }
    } else {
      transformIdStores(&BB);
    }
  }
  // for (BasicBlock *BB : forIncBlocks) {
  //   transformForInc(BB);
  // }

  transformIdLoads();
  return true;
}

llvm::PreservedAnalyses
EnforceVectorization::run(llvm::Function &F,
                          llvm::FunctionAnalysisManager &AM) {
  if (!isKernelToProcess(F))
    return llvm::PreservedAnalyses::all();

  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  if (WIH != WorkitemHandlerType::LOOPS)
    return llvm::PreservedAnalyses::all();

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
  auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);

  llvm::PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();

  EnforceVectorizationImpl WIL(DT, LI, PDT, VUA);
  // llvm::verifyFunction(F);

  return WIL.runOnFunction(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
