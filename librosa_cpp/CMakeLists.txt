cmake_minimum_required(VERSION 3.16)
project(librosa_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)

# Options
option(LIBROSA_BUILD_TESTS "Build tests" ON)
option(LIBROSA_BUILD_EXAMPLES "Build examples" OFF)
option(LIBROSA_BUILD_CROSSVAL_TESTS "Build cross-validation tests against Python librosa" OFF)
option(LIBROSA_BUILD_CLI "Build the librosa-cpp CLI tool" OFF)

# Eigen3 from submodule
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs in librosa_cpp builds" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable third-party tests in librosa_cpp builds" FORCE)
add_subdirectory(thirdparty/eigen EXCLUDE_FROM_ALL)
unset(BUILD_TESTING CACHE)
unset(BUILD_TESTING)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(FFTW3 REQUIRED fftw3)

# Optional: soxr for resampling
pkg_check_modules(SOXR soxr)

# Main library sources
set(LIBROSA_SOURCES
    # Third-party
    thirdparty/incbeta/incbeta.c

    # Utilities
    src/util/utils.cpp
    src/util/exceptions.cpp

    # Core modules
    src/core/convert.cpp
    src/core/audio.cpp
    src/core/spectrum.cpp
    src/core/pitch.cpp
    src/core/harmonic.cpp
    src/core/constantq.cpp
    src/core/notation.cpp
    src/core/intervals.cpp

    # Filters
    src/filters.cpp

    # Sequence (Viterbi, DTW)
    src/sequence.cpp

    # Feature extraction
    src/feature/spectral.cpp
    src/feature/rhythm.cpp
    src/feature/utils.cpp
    src/feature/inverse.cpp

    # High-level analysis
    src/onset.cpp
    src/beat.cpp
    src/decompose.cpp
    src/segment.cpp
    src/effects.cpp
)

# Create library
add_library(librosa ${LIBROSA_SOURCES})
add_library(librosa::librosa ALIAS librosa)

target_include_directories(librosa
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SNDFILE_INCLUDE_DIRS}
        ${FFTW3_INCLUDE_DIRS}
)

target_link_directories(librosa
    PUBLIC
        ${SNDFILE_LIBRARY_DIRS}
        ${FFTW3_LIBRARY_DIRS}
)

target_link_libraries(librosa
    PUBLIC
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        ${FFTW3_LIBRARIES}
)

if(SOXR_FOUND)
    target_compile_definitions(librosa PRIVATE LIBROSA_HAS_SOXR)
    target_include_directories(librosa PRIVATE ${SOXR_INCLUDE_DIRS})
    target_link_directories(librosa PUBLIC ${SOXR_LIBRARY_DIRS})
    target_link_libraries(librosa PUBLIC ${SOXR_LIBRARIES})
endif()

# Tests
if(LIBROSA_BUILD_TESTS)
    enable_testing()

    # Fetch Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    # Test executable
    add_executable(librosa_tests
        tests/test_main.cpp
        tests/test_util.cpp
        tests/test_convert.cpp
        tests/test_audio.cpp
        tests/test_cli.cpp
        tests/test_spectrum.cpp
        tests/test_pitch.cpp
        tests/test_constantq.cpp
        tests/test_filters.cpp
        tests/test_features.cpp
        tests/test_effects.cpp
        tests/test_sequence.cpp
        tests/test_decompose.cpp
        tests/test_onset.cpp
        tests/test_beat.cpp
        tests/test_harmonic.cpp
        tests/test_segment.cpp
        tests/test_notation.cpp
        tests/test_intervals.cpp
        cli/common.cpp
        cli/formatter.cpp
    )

    target_link_libraries(librosa_tests
        PRIVATE
            librosa::librosa
            GTest::gtest
            GTest::gtest_main
    )

    target_include_directories(librosa_tests
        PRIVATE
            ${SNDFILE_INCLUDE_DIRS}
    )

    include(GoogleTest)
    gtest_discover_tests(librosa_tests)

    # Cross-validation tests (optional, requires running generate_references.py first)
    if(LIBROSA_BUILD_CROSSVAL_TESTS)
        add_executable(crossval_tests
            tests/crossval/test_crossval.cpp
        )

        target_link_libraries(crossval_tests
            PRIVATE
                librosa::librosa
                GTest::gtest
                GTest::gtest_main
        )

        # Set working directory for finding reference data
        set_target_properties(crossval_tests PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )

        gtest_discover_tests(crossval_tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()

# Examples
if(LIBROSA_BUILD_EXAMPLES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/load_audio.cpp")
        message(FATAL_ERROR
            "LIBROSA_BUILD_EXAMPLES=ON but the examples/ sources are not present in this checkout.")
    endif()

    add_executable(example_load_audio examples/load_audio.cpp)
    target_link_libraries(example_load_audio PRIVATE librosa::librosa)

    add_executable(example_stft examples/stft_example.cpp)
    target_link_libraries(example_stft PRIVATE librosa::librosa)

    add_executable(example_mfcc examples/mfcc_example.cpp)
    target_link_libraries(example_mfcc PRIVATE librosa::librosa)
endif()

# CLI tool
if(LIBROSA_BUILD_CLI)
    include(FetchContent)
    FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.4.2
    )
    FetchContent_MakeAvailable(CLI11)

    add_executable(librosa-cpp
        cli/main.cpp
        cli/common.cpp
        cli/formatter.cpp
        cli/commands/info.cpp
        cli/commands/tempo.cpp
        cli/commands/beat.cpp
        cli/commands/onset.cpp
        cli/commands/pitch.cpp
        cli/commands/mfcc.cpp
        cli/commands/melspectrogram.cpp
        cli/commands/chroma.cpp
        cli/commands/spectral.cpp
        cli/commands/rms.cpp
        cli/commands/zcr.cpp
        cli/commands/tonnetz.cpp
        cli/commands/tuning.cpp
        cli/commands/stft.cpp
        cli/commands/trim.cpp
        cli/commands/hpss.cpp
    )

    target_link_libraries(librosa-cpp
        PRIVATE
            librosa::librosa
            CLI11::CLI11
    )

    target_include_directories(librosa-cpp
        PRIVATE
            ${SNDFILE_INCLUDE_DIRS}
    )

    install(TARGETS librosa-cpp
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS librosa
    EXPORT librosaTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/librosa
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT librosaTargets
    FILE librosaTargets.cmake
    NAMESPACE librosa::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/librosa
)
