module {
  firrtl.circuit "HardwareCounter" {
    firrtl.module @HardwareCounter(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %getCountOUT: !firrtl.sint<32>, in %getCount_EN: !firrtl.uint<1>, in %incrementEN: !firrtl.uint<1>, out %incrementRDY: !firrtl.uint<1>, in %decrementEN: !firrtl.uint<1>, out %decrementRDY: !firrtl.uint<1>, in %resetEN: !firrtl.uint<1>, out %resetRDY: !firrtl.uint<1>) {
      %count_clock, %count_reset, %count_read_data, %count_read_enable, %count_write_data, %count_write_enable = firrtl.instance count interesting_name @Register_i32_impl(in clock: !firrtl.clock, in reset: !firrtl.uint<1>, out read_data: !firrtl.uint<32>, in read_enable: !firrtl.uint<1>, in write_data: !firrtl.uint<32>, in write_enable: !firrtl.uint<1>)
      firrtl.connect %count_clock, %clock : !firrtl.clock
      firrtl.connect %count_reset, %reset : !firrtl.uint<1>
      %c1_si32 = firrtl.constant 1 : !firrtl.sint<32>
      %c1_si32_0 = firrtl.constant 1 : !firrtl.sint<32>
      %c0_si32 = firrtl.constant 0 : !firrtl.sint<32>
      %increment_wf = firrtl.node %incrementEN : !firrtl.uint<1>
      %0 = firrtl.not %increment_wf : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %1 = firrtl.and %decrementEN, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %decrement_wf = firrtl.node %1 : !firrtl.uint<1>
      %2 = firrtl.not %increment_wf : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %3 = firrtl.and %resetEN, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %4 = firrtl.not %decrement_wf : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %5 = firrtl.and %3, %4 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %reset_wf = firrtl.node %5 : !firrtl.uint<1>
      %getCount_result = firrtl.wire : !firrtl.sint<32>
      %invalid_si32 = firrtl.invalidvalue : !firrtl.sint<32>
      firrtl.connect %getCount_result, %invalid_si32 : !firrtl.sint<32>
      firrtl.connect %getCountOUT, %getCount_result : !firrtl.sint<32>
      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
      %6 = firrtl.not %decrement_wf : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %7 = firrtl.and %c1_ui1, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %8 = firrtl.not %reset_wf : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %9 = firrtl.and %7, %8 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %incrementRDY, %9 : !firrtl.uint<1>
      firrtl.when %increment_wf : !firrtl.uint<1> {
        %c1_si32_3 = firrtl.constant 1 : !firrtl.sint<32>
      }
      %c1_ui1_1 = firrtl.constant 1 : !firrtl.uint<1>
      %10 = firrtl.not %increment_wf : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %11 = firrtl.and %c1_ui1_1, %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %12 = firrtl.not %reset_wf : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %13 = firrtl.and %11, %12 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %decrementRDY, %13 : !firrtl.uint<1>
      firrtl.when %decrement_wf : !firrtl.uint<1> {
        %c1_si32_3 = firrtl.constant 1 : !firrtl.sint<32>
      }
      %c1_ui1_2 = firrtl.constant 1 : !firrtl.uint<1>
      %14 = firrtl.not %increment_wf : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %15 = firrtl.and %c1_ui1_2, %14 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %16 = firrtl.not %decrement_wf : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %17 = firrtl.and %15, %16 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      firrtl.connect %resetRDY, %17 : !firrtl.uint<1>
      firrtl.when %reset_wf : !firrtl.uint<1> {
        %c0_si32_3 = firrtl.constant 0 : !firrtl.sint<32>
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
  }
}

