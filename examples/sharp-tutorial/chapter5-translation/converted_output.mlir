sharp-opt: /home/uvxiao/sharp/.install/unified/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(const From &) [To = mlir::detail::TypedValue<circt::firrtl::BaseTypeAliasOr<circt::firrtl::IntType>>, From = mlir::OpResult]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.	Program arguments: ../../../build/bin/sharp-opt prep_output_fixed.mlir --lower-txn-body-to-firrtl
Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
0  libSharpSimulationDialect.so.21.0git 0x00007ddcf5b63d28 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) + 40
1  libSharpSimulationDialect.so.21.0git 0x00007ddcf5b61805 llvm::sys::RunSignalHandlers() + 293
2  libSharpSimulationDialect.so.21.0git 0x00007ddcf5b64421
3  libc.so.6                            0x00007ddcf1a42520
4  libc.so.6                            0x00007ddcf1a969fc pthread_kill + 300
5  libc.so.6                            0x00007ddcf1a42476 raise + 22
6  libc.so.6                            0x00007ddcf1a287f3 abort + 211
7  libc.so.6                            0x00007ddcf1a2871b
8  libc.so.6                            0x00007ddcf1a39e96
9  libSharpTxnToFIRRTL.so.21.0git       0x00007ddcf50486b4
10 libSharpTxnToFIRRTL.so.21.0git       0x00007ddcf504aee6
11 libSharpTxnToFIRRTL.so.21.0git       0x00007ddcf5059b17
12 libSharpTxnToFIRRTL.so.21.0git       0x00007ddcf5064f74
13 libSharpTxnToFIRRTL.so.21.0git       0x00007ddcf5064e88
14 libSharpSimulationPasses.so.21.0git  0x00007ddcf6927a56 mlir::ConversionPattern::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&) const + 406
15 libSharpTxnPrimitives.so.21.0git     0x00007ddcf43dccf0
16 libSharpTxnPrimitives.so.21.0git     0x00007ddcf43d9af1 mlir::PatternApplicator::matchAndRewrite(mlir::Operation*, mlir::PatternRewriter&, llvm::function_ref<bool (mlir::Pattern const&)>, llvm::function_ref<void (mlir::Pattern const&)>, llvm::function_ref<llvm::LogicalResult (mlir::Pattern const&)>) + 929
17 libSharpSimulationPasses.so.21.0git  0x00007ddcf6928a09
18 libSharpSimulationPasses.so.21.0git  0x00007ddcf6927b67 mlir::OperationConverter::convert(mlir::ConversionPatternRewriter&, mlir::Operation*) + 39
19 libSharpSimulationPasses.so.21.0git  0x00007ddcf6928c9f mlir::OperationConverter::convertOperations(llvm::ArrayRef<mlir::Operation*>) + 271
20 libSharpSimulationPasses.so.21.0git  0x00007ddcf692ee27 mlir::applyPartialConversion(mlir::Operation*, mlir::ConversionTarget const&, mlir::FrozenRewritePatternSet const&, mlir::ConversionConfig) + 55
21 libSharpTxnToFIRRTL.so.21.0git       0x00007ddcf5055d61
22 libSharpSimulationPasses.so.21.0git  0x00007ddcf68fabef mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) + 703
23 libSharpSimulationPasses.so.21.0git  0x00007ddcf68fb490 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) + 336
24 libSharpSimulationPasses.so.21.0git  0x00007ddcf68fdce2 mlir::PassManager::run(mlir::Operation*) + 898
25 sharp-opt                            0x00005c7a6c68da8a
26 sharp-opt                            0x00005c7a6c68d6f5
27 sharp-opt                            0x00005c7a6c6a2b95
28 sharp-opt                            0x00005c7a6c685ef5
29 sharp-opt                            0x00005c7a6c6861a4
30 sharp-opt                            0x00005c7a6c6863e5
31 sharp-opt                            0x00005c7a68651afa
32 libc.so.6                            0x00007ddcf1a29d90
33 libc.so.6                            0x00007ddcf1a29e40 __libc_start_main + 128
34 sharp-opt                            0x00005c7a68651965
