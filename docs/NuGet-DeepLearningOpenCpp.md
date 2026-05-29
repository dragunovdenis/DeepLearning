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
consumer's own compiler / toolset builds the library — eliminating MSVC ABI
and STL-version drift concerns inherent to prebuilt static libraries.

- Package id: **`DeepLearningOpenCpp`**
- Language standard: **C++20** (`/std:c++20 /permissive-` enforced on consumer)
- Transitive dependency: `msgpack-c-cpp-3.1.1-winsoft666` (`[1.0.0.2, )`)

## Installation

In a native C++ project (`.vcxproj`):

```powershell
Install-Package DeepLearningOpenCpp
```

or add to `packages.config`:

```xml
<package id="DeepLearningOpenCpp" version="1.0.0" targetFramework="native" />
<package id="msgpack-c-cpp-3.1.1-winsoft666" version="1.0.0.2" targetFramework="native" />
```

After NuGet restore, the package will:

1. Add `build\native\include\` to `AdditionalIncludeDirectories`.
2. Add every shipped `.cpp` to the project's `ClCompile` items (with
   `<PrecompiledHeader>NotUsing</PrecompiledHeader>`).
3. Set `<LanguageStandard>stdcpp20`, `<ConformanceMode>true`.
4. Conditionally append `USE_AVX2;` and `USE_SINGLE_PRECISION;` to
   `PreprocessorDefinitions`, and set `<EnableEnhancedInstructionSet>` to
   `AdvancedVectorExtensions2` when AVX2 is enabled.

## Opt-out properties

Both feature defines are **on** by default. To disable, set the corresponding
MSBuild property to `false` in your `.vcxproj` (in the top-level
`PropertyGroup`) or in a `Directory.Build.props`:

| Property                          | Default | Effect when `false`                              |
| --------------------------------- | ------- | ------------------------------------------------ |
| `DeepLearningUseAvx2`             | `true`  | Drops `USE_AVX2` define and `/arch:AVX2`.        |
| `DeepLearningUseSinglePrecision`  | `true`  | Drops `USE_SINGLE_PRECISION` (uses `double`).    |

Example:

```xml
<PropertyGroup Label="DeepLearningOpenCpp">
  <DeepLearningUseAvx2>false</DeepLearningUseAvx2>
  <DeepLearningUseSinglePrecision>false</DeepLearningUseSinglePrecision>
</PropertyGroup>
```

## Migrating `TrainingCell` from `<ProjectReference>` to the NuGet package

In `TrainingCell\TrainingCell.vcxproj`, remove the project reference:

```xml
<ProjectReference Include="..\DeepLearning\DeepLearning\DeepLearning.vcxproj">
  <Project>{50160b3b-7ff2-43c0-b650-54332ef4fa75}</Project>
</ProjectReference>
```

…and add the package to `packages.config` (or via `Install-Package`). The
git submodule that pulls `DeepLearning` can then be removed.

## Versioning

The package version is inferred from the most recent Git tag of the form
`v<MAJOR>.<MINOR>.<PATCH>` (e.g. `v1.0.0` → `1.0.0`). The CI workflow
publishes to nuget.org only on `v*` tag pushes; other pushes produce a CI
prerelease (`<base>-ci.<run_number>`) artifact for inspection.

## Toolset

The CI runner uses `windows-2022` with `VCToolsVersion=14.44.35207` pinned.
This is informational only — because the package is source-only the
consumer's local toolset is what compiles the code.

## Troubleshooting

- **`cl : Command line error D8016 : '/arch:AVX2' and '/arch:...' incompatible`**
  Set `<DeepLearningUseAvx2>false</DeepLearningUseAvx2>` or remove the
  conflicting `/arch:` flag from the consumer project.
- **PCH errors on injected `.cpp` files**: ensure the consumer project does
  not force `<PrecompiledHeader>Use</PrecompiledHeader>` globally via
  `<ItemDefinitionGroup>`. The package sets `NotUsing` per-file but a
  later `<ItemDefinitionGroup>` could override it.
- **Missing msgpack headers**: confirm the `msgpack-c-cpp-3.1.1-winsoft666`
  transitive dependency restored; check `packages\` directory.
