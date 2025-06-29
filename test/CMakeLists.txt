# Configure lit test suite
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SHARP_TEST_DEPENDS
  FileCheck count not
  sharp-opt
  )

add_lit_testsuite(check-sharp-lit "Running the Sharp regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SHARP_TEST_DEPENDS}
  )
set_target_properties(check-sharp-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(SHARP ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SHARP_TEST_DEPENDS})

# Create check-sharp target that includes all available tests
if(TARGET check-sharp-unit)
  add_custom_target(check-sharp
    DEPENDS check-sharp-lit check-sharp-unit
  )
else()
  add_custom_target(check-sharp
    DEPENDS check-sharp-lit
  )
endif()