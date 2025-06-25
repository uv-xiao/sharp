add_custom_target(SharpUnitTests)
set_target_properties(SharpUnitTests PROPERTIES FOLDER "Sharp Unit Tests")

function(add_sharp_unittest test_name)
  add_unittest(SharpUnitTests ${test_name} ${ARGN})
endfunction()

# Add unit test subdirectories here
# add_subdirectory(Dialect)
# add_subdirectory(Support)

add_custom_target(check-sharp-unit
  DEPENDS SharpUnitTests
)