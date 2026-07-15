# DeepLearningOpenCpp NuGet package

As it was pointed out by Michael Nielsen in his book Neural Networks and Deep Learning:

"In many parts of science – especially those parts that deal with simple phenomena – 
it’s possible to obtain very solid, very reliable evidence for quite general hypotheses. 
But in neural networks there are large numbers of parameters and hyper-parameters, 
and extremely complex interactions between them. In such extraordinarily complex systems 
it’s exceedingly difficult to establish reliable general statements. Understanding neural 
networks in their full generality is a problem that, like quantum foundations, tests the 
limits of the human mind."

This project, which was mostly inspired by the book by Michael Nielsen, is pursuing two main goals:

1. provide an open source C++ framework for developing and experimenting with neural network-based algorithms;
2. develop a high performance open source C++ library/NuGet package of machine learning algorithms.


A **source-only** native NuGet package that ships the headers (`.h`, `.inl`) and
sources (`.cpp`) of the [`DeepLearning`](https://github.com/dragunovdenis/DeepLearning)
project. The package injects them into the consumer's C++ project so the
consumer's own compiler / toolset builds the library - eliminating MSVC ABI
and STL-version drift concerns inherent to prebuilt static libraries.

- Package id: **`DeepLearningOpenCpp`**
- Language standard: **C++20** (`/std:c++20`)
- Transitive dependency: `msgpack-c-cpp-3.1.1-winsoft666` (`[1.0.0.2, )`)

## Shipped files

The package contains 63 headers (`.h`), 14 inline files (`.inl`) and 26 source
files (`.cpp`) organised under `build\native\include\` and `build\native\src\`.

## Installation

In a native C++ project (`.vcxproj`):

```powershell
Install-Package DeepLearningOpenCpp
```

or add directly to `packages.config`:

```xml
<package id="DeepLearningOpenCpp" version="2.6.0" targetFramework="native" />
<package id="msgpack-c-cpp-3.1.1-winsoft666" version="1.0.0.2" targetFramework="native" />
```

After NuGet restore, the package will:

1. Add the include root `build\native\include\` plus the sub-folder roots
   `include\Math\`, `include\NeuralNet\`, `include\Diagnostics\`,
   `include\ImageProcessing\` and `include\ThirdParty\` to
   `AdditionalIncludeDirectories`. The sub-folder entries are required so
   that bare-name sibling includes inside the shipped `.cpp` sources resolve
   without any extra configuration on the consumer side.
2. Set `<LanguageStandard>stdcpp20</LanguageStandard>` on all compiled items.
3. Unless `DeepLearningCompileSources` is set to `false`, add every shipped
   `.cpp` to the project's `ClCompile` items (with
   `<PrecompiledHeader>NotUsing</PrecompiledHeader>`).

## `DeepLearningCompileSources` property

The single public MSBuild property exposed by the package. Override it in a
`<PropertyGroup>` of your `.vcxproj` **before** the `Microsoft.Cpp.Default.props`
import (or in a `Directory.Build.props`):

| Property                     | Default | Effect when `false`                                              |
| ---------------------------- | ------- | ---------------------------------------------------------------- |
| `DeepLearningCompileSources` | `true`  | Skips injecting the shipped `.cpp` files (header-only consumer). |

## `USE_AVX2` and `USE_SINGLE_PRECISION`

The library sources respond to two preprocessor symbols that the consumer
project is responsible for defining (the package does **not** inject them):

- **`USE_AVX2`** - enables AVX2 SIMD intrinsics in the math kernel. You should
  also set `<EnableEnhancedInstructionSet>AdvancedVectorExtensions2</EnableEnhancedInstructionSet>`
  so the compiler emits AVX2 instructions across the whole translation unit.
  Omitting this symbol falls back to scalar code that runs on any x64 CPU.
- **`USE_SINGLE_PRECISION`** - makes `DeepLearning::Real` an alias for `float`
  (32-bit). When absent, `Real` is `double` (64-bit). All tensors, weight
  matrices and activation values follow this typedef, so **the symbol must be
  defined consistently across every translation unit that includes DeepLearning
  headers**, including the consumer's own `.cpp` files.

Define them inside the appropriate per-configuration `<ItemDefinitionGroup>`:

```xml
<!-- AVX2 + float for a Release configuration -->
<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
  <ClCompile>
    <PreprocessorDefinitions>USE_AVX2;USE_SINGLE_PRECISION;%(PreprocessorDefinitions)</PreprocessorDefinitions>
    <EnableEnhancedInstructionSet>AdvancedVectorExtensions2</EnableEnhancedInstructionSet>
  </ClCompile>
</ItemDefinitionGroup>

<!-- No AVX2, double precision for a Debug configuration -->
<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
  <ClCompile>
    <PreprocessorDefinitions>%(PreprocessorDefinitions)</PreprocessorDefinitions>
  </ClCompile>
</ItemDefinitionGroup>
```

> **Important:** every project in the solution that includes DeepLearning
> headers must define `USE_SINGLE_PRECISION` (or not) **consistently** with the
> project that compiled the sources. A mismatch causes `LNK2019` unresolved
> external errors because the template specialisations in the `.lib` were
> compiled against a different `Real` type.

## Multi-project solutions (header-only consumers)

In a solution where one project (e.g. a static-lib `Core`) compiles the
DeepLearning sources and other projects (test DLL, wrapper DLL, executable)
consume `Core` transitively, **only `Core` should compile the sources**.
Otherwise the sources are compiled into every project and the linker reports
`LNK2005` duplicate-symbol errors.

For each header-only consumer project:

1. Install the package (so NuGet adds the include directories and language
   standard settings).
2. Set `DeepLearningCompileSources=false`:

```xml
<PropertyGroup Label="DeepLearningOpenCpp">
  <DeepLearningCompileSources>false</DeepLearningCompileSources>
</PropertyGroup>
```

> **Note:** `DeepLearningCompileSources=false` is only safe when the upstream
> `.lib` already exports every DeepLearning symbol the header-only consumer
> uses. If the consumer calls template specialisations or free functions that
> the `.lib` project never instantiates, the linker will still report `LNK2019`.
> In that case keep `DeepLearningCompileSources=true` (the default) so those
> symbols are compiled locally.

## Release history

### 2.6.0
- **Bug fix – `ThreadPool`: out-of-bounds `context_data` access during training.**
  Worker threads in the pool are assigned a fixed local ID in `[0, N_threads-1]`.
  Previously, each job received that *thread* ID and used it to index a
  per-job context array sized to the number of jobs in the current mini-batch
  (`M ≤ N_threads`). There were no mechanism to ensure that the next launched 
  job will receive the lowest *thread* ID out of available ones. Whenever the 
  thread that picked up a job had an ID ≥ M the access was out of bounds, 
  causing silent gradient corruption or crashes. The fix captures a sequential
  *job counter* (0 … M-1) by value into each lambda, decoupling the data-slot 
  index from the worker-thread identity. Companion clean-ups: the pool now 
  tracks queued-job count internally so callers no longer supply an 
  expected-completion count to `wait_all_jobs_done()` (replaces the former 
  `wait_until_jobs_done(expected_count)`), job counters are typed as `std::size_t`, 
  and `notify_all` is replaced with `notify_one`.
- **Fully deterministic implementation of `Net<D>.learn(...)`**
  The multi-threaded batch processing was reworked so that now gradient 
  accumulation happens independently in each thread instead of using a
  centralized accumulation container. This allowed to get rid of the mutex
  synchronization which seems to have a considerable positive impact on
  the performance of the method.
- `class CumulativeGradient` was removed as it is no longer used.

### 2.5.0
- Full AVX2 support for the `Matrix` class.
- Switch to Intel C++ Compiler 2026.
- Solution and regression tests migrated to Visual Studio 2026.

## Versioning

The package version is inferred from the most recent Git tag of the form
`v<MAJOR>.<MINOR>.<PATCH>` (e.g. `v1.0.0` -> `1.0.0`). The CI workflow
publishes to nuget.org only on `v*` tag pushes; other pushes produce a CI
prerelease (`<base>-ci.<run_number>`) artifact for inspection.

## Toolset

The CI runner uses `windows-2022` with `VCToolsVersion=14.44.35207` pinned.
This is informational only - because the package is source-only the
consumer's local toolset is what compiles the code.

## Troubleshooting

- **`LNK2019` / unresolved externals after setting `DeepLearningCompileSources=false`**:
  The upstream `.lib` does not export a symbol the consumer needs. Either keep
  `DeepLearningCompileSources=true`, or ensure the `.lib` project uses the
  symbol itself so the specialisation is instantiated.
- **`LNK2019` with `MemHandleConst<float>` / `MemHandleConst<double>` mismatch**:
  `USE_SINGLE_PRECISION` is inconsistent between the project that compiled the
  `.lib` and the consumer. Align the define across all projects.
- **`LNK2005` / "already defined" errors in a multi-project solution**:
  The shipped `.cpp` sources are being compiled into more than one project.
  Pick a single project to compile them and set
  `<DeepLearningCompileSources>false</DeepLearningCompileSources>` on the
  others (see *Multi-project solutions* above).
- **`cl : Command line error D8016 : '/arch:AVX2' and '/arch:...' incompatible`**:
  Remove the conflicting `/arch:` flag from your project or drop
  `<EnableEnhancedInstructionSet>AdvancedVectorExtensions2</EnableEnhancedInstructionSet>`
  from the configuration that triggers it.
- **PCH errors on injected `.cpp` files**: ensure the consumer project does
  not force `<PrecompiledHeader>Use</PrecompiledHeader>` globally via
  `<ItemDefinitionGroup>`. The package sets `NotUsing` per-file, but a
  later `<ItemDefinitionGroup>` can override it.
- **Missing msgpack headers**: confirm the `msgpack-c-cpp-3.1.1-winsoft666`
  transitive dependency was restored; check the `packages\` directory.