; ModuleID = '/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/hello_world/proj_simple_add/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

%"struct.ap_int<8>" = type { %"struct.ap_int_base<8, true>" }
%"struct.ap_int_base<8, true>" = type { %"struct.ssdm_int<8, true>" }
%"struct.ssdm_int<8, true>" = type { i8 }
%"struct.ap_int<16>" = type { %"struct.ap_int_base<16, true>" }
%"struct.ap_int_base<16, true>" = type { %"struct.ssdm_int<16, true>" }
%"struct.ssdm_int<16, true>" = type { i16 }

; Function Attrs: inaccessiblemem_or_argmemonly noinline willreturn
define void @apatb_simple_add_ir(%"struct.ap_int<8>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="128" %a, %"struct.ap_int<8>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="128" %b, %"struct.ap_int<16>"* noalias nocapture nonnull "fpga.decayed.dim.hint"="128" %c) local_unnamed_addr #0 {
entry:
  %a_copy = alloca [128 x i8], align 512
  %b_copy = alloca [128 x i8], align 512
  %c_copy = alloca [128 x i16], align 512
  %0 = bitcast %"struct.ap_int<8>"* %a to [128 x %"struct.ap_int<8>"]*
  %1 = bitcast %"struct.ap_int<8>"* %b to [128 x %"struct.ap_int<8>"]*
  %2 = bitcast %"struct.ap_int<16>"* %c to [128 x %"struct.ap_int<16>"]*
  call fastcc void @copy_in([128 x %"struct.ap_int<8>"]* nonnull %0, [128 x i8]* nonnull align 512 %a_copy, [128 x %"struct.ap_int<8>"]* nonnull %1, [128 x i8]* nonnull align 512 %b_copy, [128 x %"struct.ap_int<16>"]* nonnull %2, [128 x i16]* nonnull align 512 %c_copy)
  call void @apatb_simple_add_hw([128 x i8]* %a_copy, [128 x i8]* %b_copy, [128 x i16]* %c_copy)
  call void @copy_back([128 x %"struct.ap_int<8>"]* %0, [128 x i8]* %a_copy, [128 x %"struct.ap_int<8>"]* %1, [128 x i8]* %b_copy, [128 x %"struct.ap_int<16>"]* %2, [128 x i16]* %c_copy)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_in([128 x %"struct.ap_int<8>"]* noalias readonly "unpacked"="0", [128 x i8]* noalias nocapture align 512 "unpacked"="1.0", [128 x %"struct.ap_int<8>"]* noalias readonly "unpacked"="2", [128 x i8]* noalias nocapture align 512 "unpacked"="3.0", [128 x %"struct.ap_int<16>"]* noalias readonly "unpacked"="4", [128 x i16]* noalias nocapture align 512 "unpacked"="5.0") unnamed_addr #1 {
entry:
  call fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<8>"([128 x i8]* align 512 %1, [128 x %"struct.ap_int<8>"]* %0)
  call fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<8>"([128 x i8]* align 512 %3, [128 x %"struct.ap_int<8>"]* %2)
  call fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<16>"([128 x i16]* align 512 %5, [128 x %"struct.ap_int<16>"]* %4)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<16>"([128 x i16]* noalias nocapture align 512 "unpacked"="0.0" %dst, [128 x %"struct.ap_int<16>"]* noalias readonly "unpacked"="1" %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [128 x %"struct.ap_int<16>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a128struct.ap_int<16>"([128 x i16]* %dst, [128 x %"struct.ap_int<16>"]* nonnull %src, i64 128)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a128struct.ap_int<16>"([128 x i16]* nocapture "unpacked"="0.0" %dst, [128 x %"struct.ap_int<16>"]* readonly "unpacked"="1" %src, i64 "unpacked"="2" %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [128 x %"struct.ap_int<16>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [128 x %"struct.ap_int<16>"], [128 x %"struct.ap_int<16>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %dst.addr.0.0.06 = getelementptr [128 x i16], [128 x i16]* %dst, i64 0, i64 %for.loop.idx2
  %1 = load i16, i16* %src.addr.0.0.05, align 2
  store i16 %1, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_out([128 x %"struct.ap_int<8>"]* noalias "unpacked"="0", [128 x i8]* noalias nocapture readonly align 512 "unpacked"="1.0", [128 x %"struct.ap_int<8>"]* noalias "unpacked"="2", [128 x i8]* noalias nocapture readonly align 512 "unpacked"="3.0", [128 x %"struct.ap_int<16>"]* noalias "unpacked"="4", [128 x i16]* noalias nocapture readonly align 512 "unpacked"="5.0") unnamed_addr #4 {
entry:
  call fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<8>.22"([128 x %"struct.ap_int<8>"]* %0, [128 x i8]* align 512 %1)
  call fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<8>.22"([128 x %"struct.ap_int<8>"]* %2, [128 x i8]* align 512 %3)
  call fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<16>.5"([128 x %"struct.ap_int<16>"]* %4, [128 x i16]* align 512 %5)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<16>.5"([128 x %"struct.ap_int<16>"]* noalias "unpacked"="0" %dst, [128 x i16]* noalias nocapture readonly align 512 "unpacked"="1.0" %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [128 x %"struct.ap_int<16>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a128struct.ap_int<16>.8"([128 x %"struct.ap_int<16>"]* nonnull %dst, [128 x i16]* %src, i64 128)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a128struct.ap_int<16>.8"([128 x %"struct.ap_int<16>"]* "unpacked"="0" %dst, [128 x i16]* nocapture readonly "unpacked"="1.0" %src, i64 "unpacked"="2" %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [128 x %"struct.ap_int<16>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [128 x i16], [128 x i16]* %src, i64 0, i64 %for.loop.idx2
  %dst.addr.0.0.06 = getelementptr [128 x %"struct.ap_int<16>"], [128 x %"struct.ap_int<16>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = load i16, i16* %src.addr.0.0.05, align 2
  store i16 %1, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<8>"([128 x i8]* noalias nocapture align 512 "unpacked"="0.0" %dst, [128 x %"struct.ap_int<8>"]* noalias readonly "unpacked"="1" %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [128 x %"struct.ap_int<8>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a128struct.ap_int<8>.18"([128 x i8]* %dst, [128 x %"struct.ap_int<8>"]* nonnull %src, i64 128)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a128struct.ap_int<8>.18"([128 x i8]* nocapture "unpacked"="0.0" %dst, [128 x %"struct.ap_int<8>"]* readonly "unpacked"="1" %src, i64 "unpacked"="2" %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [128 x %"struct.ap_int<8>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [128 x %"struct.ap_int<8>"], [128 x %"struct.ap_int<8>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %dst.addr.0.0.06 = getelementptr [128 x i8], [128 x i8]* %dst, i64 0, i64 %for.loop.idx2
  %1 = load i8, i8* %src.addr.0.0.05, align 1
  store i8 %1, i8* %dst.addr.0.0.06, align 1
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<8>.22"([128 x %"struct.ap_int<8>"]* noalias "unpacked"="0" %dst, [128 x i8]* noalias nocapture readonly align 512 "unpacked"="1.0" %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [128 x %"struct.ap_int<8>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a128struct.ap_int<8>.25"([128 x %"struct.ap_int<8>"]* nonnull %dst, [128 x i8]* %src, i64 128)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a128struct.ap_int<8>.25"([128 x %"struct.ap_int<8>"]* "unpacked"="0" %dst, [128 x i8]* nocapture readonly "unpacked"="1.0" %src, i64 "unpacked"="2" %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [128 x %"struct.ap_int<8>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [128 x i8], [128 x i8]* %src, i64 0, i64 %for.loop.idx2
  %dst.addr.0.0.06 = getelementptr [128 x %"struct.ap_int<8>"], [128 x %"struct.ap_int<8>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = load i8, i8* %src.addr.0.0.05, align 1
  store i8 %1, i8* %dst.addr.0.0.06, align 1
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

declare void @apatb_simple_add_hw([128 x i8]*, [128 x i8]*, [128 x i16]*)

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_back([128 x %"struct.ap_int<8>"]* noalias "unpacked"="0", [128 x i8]* noalias nocapture readonly align 512 "unpacked"="1.0", [128 x %"struct.ap_int<8>"]* noalias "unpacked"="2", [128 x i8]* noalias nocapture readonly align 512 "unpacked"="3.0", [128 x %"struct.ap_int<16>"]* noalias "unpacked"="4", [128 x i16]* noalias nocapture readonly align 512 "unpacked"="5.0") unnamed_addr #4 {
entry:
  call fastcc void @"onebyonecpy_hls.p0a128struct.ap_int<16>.5"([128 x %"struct.ap_int<16>"]* %4, [128 x i16]* align 512 %5)
  ret void
}

define void @simple_add_hw_stub_wrapper([128 x i8]*, [128 x i8]*, [128 x i16]*) #5 {
entry:
  %3 = alloca [128 x %"struct.ap_int<8>"]
  %4 = alloca [128 x %"struct.ap_int<8>"]
  %5 = alloca [128 x %"struct.ap_int<16>"]
  call void @copy_out([128 x %"struct.ap_int<8>"]* %3, [128 x i8]* %0, [128 x %"struct.ap_int<8>"]* %4, [128 x i8]* %1, [128 x %"struct.ap_int<16>"]* %5, [128 x i16]* %2)
  %6 = bitcast [128 x %"struct.ap_int<8>"]* %3 to %"struct.ap_int<8>"*
  %7 = bitcast [128 x %"struct.ap_int<8>"]* %4 to %"struct.ap_int<8>"*
  %8 = bitcast [128 x %"struct.ap_int<16>"]* %5 to %"struct.ap_int<16>"*
  call void @simple_add_hw_stub(%"struct.ap_int<8>"* %6, %"struct.ap_int<8>"* %7, %"struct.ap_int<16>"* %8)
  call void @copy_in([128 x %"struct.ap_int<8>"]* %3, [128 x i8]* %0, [128 x %"struct.ap_int<8>"]* %4, [128 x i8]* %1, [128 x %"struct.ap_int<16>"]* %5, [128 x i16]* %2)
  ret void
}

declare void @simple_add_hw_stub(%"struct.ap_int<8>"*, %"struct.ap_int<8>"*, %"struct.ap_int<16>"*)

attributes #0 = { inaccessiblemem_or_argmemonly noinline willreturn "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyin" }
attributes #2 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #3 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="arraycpy_hls" }
attributes #4 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyout" }
attributes #5 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
