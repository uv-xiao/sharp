add_custom_target(SharpUnitTests)
set_target_properties(SharpUnitTests PROPERTIES FOLDER "Sharp Unit Tests")

function(add_sharp_unittest test_name)
  add_unittest(SharpUnitTests ${test_name} ${ARGN})
endfunction()

# Add unit test subdirectories here
add_subdirectory(Dialect)
add_subdirectory(Simulation)
# add_subdirectory(Support)

# Create a shell script that finds and runs all test executables
set(RUN_UNIT_TESTS_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/run-all-unit-tests.sh)
file(WRITE ${RUN_UNIT_TESTS_SCRIPT} 
"#!/bin/bash
# Script to run all Sharp unit tests
echo 'Running Sharp unit tests...'
failed=0
total=0

# Find all test executables in the unittests directory
for test_exe in \$(find ${CMAKE_CURRENT_BINARY_DIR} -name '*Test' -type f -executable); do
  test_name=\$(basename \$test_exe)
  echo \"Running \$test_name...\"
  if \$test_exe; then
    echo \"✓ \$test_name passed\"
  else
    echo \"✗ \$test_name failed\"
    failed=\$((failed + 1))
  fi
  total=\$((total + 1))
  echo
done

if [ \$total -eq 0 ]; then
  echo \"No unit tests found!\"
  exit 1
fi

echo \"Unit test summary: \$((total - failed))/\$total passed\"

if [ \$failed -gt 0 ]; then
  exit 1
else
  exit 0
fi
")

# Make script executable
execute_process(COMMAND chmod +x ${RUN_UNIT_TESTS_SCRIPT})

# Create the check target that runs the script
add_custom_target(check-sharp-unit
  COMMAND ${RUN_UNIT_TESTS_SCRIPT}
  DEPENDS SharpUnitTests
  COMMENT "Running Sharp unit tests"
  USES_TERMINAL
)