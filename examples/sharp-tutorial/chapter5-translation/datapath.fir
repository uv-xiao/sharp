module {
  firrtl.circuit "Datapath" {
    firrtl.module @Datapath(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %pushDataOUT: !firrtl.sint<32>, in %pushDataEN: !firrtl.uint<1>, out %pushDataRDY: !firrtl.uint<1>, in %processEN: !firrtl.uint<1>, out %processRDY: !firrtl.uint<1>, out %getOutputOUT: !firrtl.sint<32>, in %getOutput_EN: !firrtl.uint<1>, out %isReadyOUT: !firrtl.uint<1>, in %isReady_EN: !firrtl.uint<1>) {
      %output_reg_clock, %output_reg_reset, %output_reg_read_data, %output_reg_read_enable, %output_reg_write_data, %output_reg_write_enable = firrtl.instance output_reg interesting_name @Register_i32_impl(in clock: !firrtl.clock, in reset: !firrtl.uint<1>, out read_data: !firrtl.uint<32>, in read_enable: !firrtl.uint<1>, in write_data: !firrtl.uint<32>, in write_enable: !firrtl.uint<1>)
      firrtl.connect %output_reg_clock, %clock : !firrtl.clock
      firrtl.connect %output_reg_reset, %reset : !firrtl.uint<1>
      %status_clock, %status_reset, %status_read_data, %status_read_enable, %status_write_data, %status_write_enable = firrtl.instance status interesting_name @Register_i1_impl(in clock: !firrtl.clock, in reset: !firrtl.uint<1>, out read_data: !firrtl.uint<1>, in read_enable: !firrtl.uint<1>, in write_data: !firrtl.uint<1>, in write_enable: !firrtl.uint<1>)
      firrtl.connect %status_clock, %clock : !firrtl.clock
      firrtl.connect %status_reset, %reset : !firrtl.uint<1>
      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
      %c2_si32 = firrtl.constant 2 : !firrtl.sint<32>
      %pushData_wf = firrtl.node %pushDataEN : !firrtl.uint<1>
      %process_wf = firrtl.node %processEN : !firrtl.uint<1>
      %getOutput_result = firrtl.wire : !firrtl.sint<32>
      %invalid_si32 = firrtl.invalidvalue : !firrtl.sint<32>
      firrtl.connect %getOutput_result, %invalid_si32 : !firrtl.sint<32>
      firrtl.connect %getOutputOUT, %getOutput_result : !firrtl.sint<32>
      %isReady_result = firrtl.wire : !firrtl.uint<1>
      %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
      firrtl.connect %isReady_result, %invalid_ui1 : !firrtl.uint<1>
      firrtl.connect %isReadyOUT, %isReady_result : !firrtl.uint<1>
      %c1_ui1_0 = firrtl.constant 1 : !firrtl.uint<1>
      firrtl.connect %pushDataRDY, %c1_ui1_0 : !firrtl.uint<1>
      firrtl.when %pushData_wf : !firrtl.uint<1> {
      }
      %c1_ui1_1 = firrtl.constant 1 : !firrtl.uint<1>
      firrtl.connect %processRDY, %c1_ui1_1 : !firrtl.uint<1>
      firrtl.when %process_wf : !firrtl.uint<1> {
        %c1_ui1_2 = firrtl.constant 1 : !firrtl.uint<1>
        %c2_si32_3 = firrtl.constant 2 : !firrtl.sint<32>
      }
    }
    firrtl.module @Register_i32_impl(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %read_data: !firrtl.uint<32>, in %read_enable: !firrtl.uint<1>, in %write_data: !firrtl.uint<32>, in %write_enable: !firrtl.uint<1>) {
      %c0_ui32 = firrtl.constant 0 : !firrtl.uint<32>
      %reg = firrtl.regreset %clock, %reset, %c0_ui32 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>
      %0 = firrtl.not %reset : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %1 = firrtl.and %write_enable, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.when %1 : !firrtl.uint<1> {
        firrtl.connect %reg, %write_data : !firrtl.uint<32>
      }
      firrtl.connect %read_data, %reg : !firrtl.uint<32>
    }
    firrtl.module @Register_i1_impl(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %read_data: !firrtl.uint<1>, in %read_enable: !firrtl.uint<1>, in %write_data: !firrtl.uint<1>, in %write_enable: !firrtl.uint<1>) {
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      %reg = firrtl.regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
      %0 = firrtl.not %reset : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %1 = firrtl.and %write_enable, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.when %1 : !firrtl.uint<1> {
        firrtl.connect %reg, %write_data : !firrtl.uint<1>
      }
      firrtl.connect %read_data, %reg : !firrtl.uint<1>
    }
  }
}

