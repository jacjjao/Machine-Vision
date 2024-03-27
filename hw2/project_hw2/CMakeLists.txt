cmake_minimum_required(VERSION 3.15)
project(hw2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set("OpenCV_DIR" ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc14/lib)
find_package( OpenCV REQUIRED )

add_executable(${PROJECT_NAME} 
	main.cpp 
	UnionFind.cpp
)

target_include_directories(${PROJECT_NAME}
PRIVATE 
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/include
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
)

target_compile_definitions(${PROJECT_NAME}
PRIVATE 
	PROJECT_DIR="${CMAKE_SOURCE_DIR}/"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
	COMMAND_EXPAND_LISTS
)