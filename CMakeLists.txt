cmake_minimum_required(VERSION 3.24)

project(PhysTwin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PHYSTWIN_BUILD_TESTS "Build CTest targets" ON)

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(phystwin_core
    cpp/src/io.cpp
    cpp/src/simulator.cpp
    cpp/src/synthetic.cpp
    cpp/src/fitter.cpp
    cpp/src/metrics.cpp
)
target_include_directories(phystwin_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_link_libraries(phystwin_core
    PUBLIC
        nlohmann_json::nlohmann_json
)
target_compile_features(phystwin_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(phystwin_core PRIVATE /W4 /permissive- /EHsc /utf-8)
else()
    target_compile_options(phystwin_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(phystwin cpp/src/main.cpp)
target_link_libraries(phystwin PRIVATE phystwin_core)
if(MSVC)
    target_compile_options(phystwin PRIVATE /W4 /permissive- /EHsc /utf-8)
endif()

if(PHYSTWIN_BUILD_TESTS)
    enable_testing()
    include(CTest)

    add_executable(phystwin_io_test cpp/tests/io_roundtrip_test.cpp)
    target_link_libraries(phystwin_io_test PRIVATE phystwin_core)
    add_test(NAME io_roundtrip COMMAND phystwin_io_test)

    add_executable(phystwin_synthetic_fit_test cpp/tests/synthetic_fit_test.cpp)
    target_link_libraries(phystwin_synthetic_fit_test PRIVATE phystwin_core)
    add_test(NAME synthetic_fit COMMAND phystwin_synthetic_fit_test)

    add_executable(phystwin_dropped_frame_test cpp/tests/dropped_frame_test.cpp)
    target_link_libraries(phystwin_dropped_frame_test PRIVATE phystwin_core)
    add_test(NAME dropped_frame COMMAND phystwin_dropped_frame_test)
endif()
