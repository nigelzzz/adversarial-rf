; ModuleID = '/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/src/proj_awn_int8/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.sideeffect() #0

; Function Attrs: noinline willreturn
define void @apatb_awn_forward_ir([128 x i8]* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="2" %x_q, i8* noalias nocapture nonnull "fpga.decayed.dim.hint"="11" %logits_q) local_unnamed_addr #1 {
entry:
  %x_q_copy_0 = alloca [128 x i8], align 512
  %x_q_copy_1 = alloca [128 x i8], align 512
  %logits_q_copy = alloca [11 x i8], align 512
  %0 = bitcast [128 x i8]* %x_q to [2 x [128 x i8]]*
  %1 = bitcast i8* %logits_q to [11 x i8]*
  call void @copy_in([2 x [128 x i8]]* nonnull %0, [128 x i8]* nonnull align 512 %x_q_copy_0, [128 x i8]* nonnull align 512 %x_q_copy_1, [11 x i8]* nonnull %1, [11 x i8]* nonnull align 512 %logits_q_copy)
  call void @llvm.sideeffect() #8 [ "xlx_array_partition"([128 x i8]* %x_q_copy_0, i32 998, i32 1, i32 0, i1 false) ], !dbg !13
  call void @llvm.sideeffect() #8 [ "xlx_array_partition"([128 x i8]* %x_q_copy_1, i32 998, i32 1, i32 0, i1 false) ], !dbg !13
  call void @apatb_awn_forward_hw([128 x i8]* %x_q_copy_0, [128 x i8]* %x_q_copy_1, [11 x i8]* %logits_q_copy)
  call void @copy_back([2 x [128 x i8]]* %0, [128 x i8]* %x_q_copy_0, [128 x i8]* %x_q_copy_1, [11 x i8]* %1, [11 x i8]* %logits_q_copy)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a2a128i8([2 x [128 x i8]]* "orig.arg.no"="0" %dst, [2 x [128 x i8]]* readonly "orig.arg.no"="1" %src, i64 "orig.arg.no"="2" %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [2 x [128 x i8]]* %src, null
  %1 = icmp eq [2 x [128 x i8]]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [2 x [128 x i8]], [2 x [128 x i8]]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [2 x [128 x i8]], [2 x [128 x i8]]* %src, i64 0, i64 %for.loop.idx2
  call void @arraycpy_hls.p0a128i8([128 x i8]* %dst.addr, [128 x i8]* %src.addr, i64 128)
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a128i8([128 x i8]* %dst, [128 x i8]* readonly %src, i64 %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [128 x i8]* %src, null
  %1 = icmp eq [128 x i8]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [128 x i8], [128 x i8]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [128 x i8], [128 x i8]* %src, i64 0, i64 %for.loop.idx2
  %3 = load i8, i8* %src.addr, align 1
  store i8 %3, i8* %dst.addr, align 1
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a11i8([11 x i8]* noalias align 512 %dst, [11 x i8]* noalias readonly %src) unnamed_addr #3 {
entry:
  %0 = icmp eq [11 x i8]* %dst, null
  %1 = icmp eq [11 x i8]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a11i8([11 x i8]* nonnull %dst, [11 x i8]* nonnull %src, i64 11)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a11i8([11 x i8]* %dst, [11 x i8]* readonly %src, i64 %num) local_unnamed_addr #2 {
entry:
  %0 = icmp eq [11 x i8]* %src, null
  %1 = icmp eq [11 x i8]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [11 x i8], [11 x i8]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [11 x i8], [11 x i8]* %src, i64 0, i64 %for.loop.idx2
  %3 = load i8, i8* %src.addr, align 1
  store i8 %3, i8* %dst.addr, align 1
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1) #4

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a2a128i8.17.18([128 x i8]* "orig.arg.no"="0" "unpacked"="0.0" %dst_0, [128 x i8]* "orig.arg.no"="0" "unpacked"="0.1" %dst_1, [2 x [128 x i8]]* readonly "orig.arg.no"="1" %src, i64 "orig.arg.no"="2" %num) #2 {
entry:
  %0 = icmp eq [2 x [128 x i8]]* %src, null
  %1 = icmp eq [128 x i8]* %dst_0, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %dst.addr.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %dst.addr.exit ]
  %3 = trunc i64 %for.loop.idx2 to i1
  %src.addr = getelementptr [2 x [128 x i8]], [2 x [128 x i8]]* %src, i64 0, i64 %for.loop.idx2
  %cond = icmp eq i1 %3, false
  br i1 %cond, label %dst.addr.case.0, label %dst.addr.case.1

dst.addr.case.0:                                  ; preds = %for.loop
  call void @arraycpy_hls.p0a128i8([128 x i8]* %dst_0, [128 x i8]* %src.addr, i64 128)
  br label %dst.addr.exit

dst.addr.case.1:                                  ; preds = %for.loop
  call void @llvm.assume(i1 %3)
  call void @arraycpy_hls.p0a128i8([128 x i8]* %dst_1, [128 x i8]* %src.addr, i64 128)
  br label %dst.addr.exit

dst.addr.exit:                                    ; preds = %dst.addr.case.1, %dst.addr.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %dst.addr.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @onebyonecpy_hls.p0a2a128i8.16.19([128 x i8]* noalias align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst_0, [128 x i8]* noalias align 512 "orig.arg.no"="0" "unpacked"="0.1" %dst_1, [2 x [128 x i8]]* noalias readonly "orig.arg.no"="1" %src) #3 {
entry:
  %0 = icmp eq [128 x i8]* %dst_0, null
  %1 = icmp eq [2 x [128 x i8]]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a2a128i8.17.18([128 x i8]* nonnull %dst_0, [128 x i8]* %dst_1, [2 x [128 x i8]]* nonnull %src, i64 2)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_in([2 x [128 x i8]]* noalias readonly "orig.arg.no"="0", [128 x i8]* noalias align 512 "orig.arg.no"="1" "unpacked"="1.0" %_0, [128 x i8]* noalias align 512 "orig.arg.no"="1" "unpacked"="1.1" %_1, [11 x i8]* noalias readonly "orig.arg.no"="2", [11 x i8]* noalias align 512 "orig.arg.no"="3") #5 {
entry:
  call void @onebyonecpy_hls.p0a2a128i8.16.19([128 x i8]* align 512 %_0, [128 x i8]* align 512 %_1, [2 x [128 x i8]]* %0)
  call fastcc void @onebyonecpy_hls.p0a11i8([11 x i8]* align 512 %2, [11 x i8]* %1)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a2a128i8.27.28([2 x [128 x i8]]* "orig.arg.no"="0" %dst, [128 x i8]* readonly "orig.arg.no"="1" "unpacked"="1.0" %src_0, [128 x i8]* readonly "orig.arg.no"="1" "unpacked"="1.1" %src_1, i64 "orig.arg.no"="2" %num) #2 {
entry:
  %0 = icmp eq [128 x i8]* %src_0, null
  %1 = icmp eq [2 x [128 x i8]]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %src.addr.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %src.addr.exit ]
  %3 = trunc i64 %for.loop.idx2 to i1
  %dst.addr = getelementptr [2 x [128 x i8]], [2 x [128 x i8]]* %dst, i64 0, i64 %for.loop.idx2
  %cond = icmp eq i1 %3, false
  br i1 %cond, label %src.addr.case.0, label %src.addr.case.1

src.addr.case.0:                                  ; preds = %for.loop
  call void @arraycpy_hls.p0a128i8([128 x i8]* %dst.addr, [128 x i8]* %src_0, i64 128)
  br label %src.addr.exit

src.addr.case.1:                                  ; preds = %for.loop
  call void @llvm.assume(i1 %3)
  call void @arraycpy_hls.p0a128i8([128 x i8]* %dst.addr, [128 x i8]* %src_1, i64 128)
  br label %src.addr.exit

src.addr.exit:                                    ; preds = %src.addr.case.1, %src.addr.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %src.addr.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @onebyonecpy_hls.p0a2a128i8.26.29([2 x [128 x i8]]* noalias "orig.arg.no"="0" %dst, [128 x i8]* noalias readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src_0, [128 x i8]* noalias readonly align 512 "orig.arg.no"="1" "unpacked"="1.1" %src_1) #3 {
entry:
  %0 = icmp eq [2 x [128 x i8]]* %dst, null
  %1 = icmp eq [128 x i8]* %src_0, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a2a128i8.27.28([2 x [128 x i8]]* nonnull %dst, [128 x i8]* nonnull %src_0, [128 x i8]* %src_1, i64 2)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_out([2 x [128 x i8]]* noalias "orig.arg.no"="0", [128 x i8]* noalias readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %_0, [128 x i8]* noalias readonly align 512 "orig.arg.no"="1" "unpacked"="1.1" %_1, [11 x i8]* noalias "orig.arg.no"="2", [11 x i8]* noalias readonly align 512 "orig.arg.no"="3") #6 {
entry:
  call void @onebyonecpy_hls.p0a2a128i8.26.29([2 x [128 x i8]]* %0, [128 x i8]* align 512 %_0, [128 x i8]* align 512 %_1)
  call fastcc void @onebyonecpy_hls.p0a11i8([11 x i8]* %1, [11 x i8]* align 512 %2)
  ret void
}

declare void @apatb_awn_forward_hw([128 x i8]*, [128 x i8]*, [11 x i8]*)

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_back([2 x [128 x i8]]* noalias "orig.arg.no"="0", [128 x i8]* noalias readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %_0, [128 x i8]* noalias readonly align 512 "orig.arg.no"="1" "unpacked"="1.1" %_1, [11 x i8]* noalias "orig.arg.no"="2", [11 x i8]* noalias readonly align 512 "orig.arg.no"="3") #6 {
entry:
  call fastcc void @onebyonecpy_hls.p0a11i8([11 x i8]* %1, [11 x i8]* align 512 %2)
  ret void
}

define void @awn_forward_hw_stub_wrapper([128 x i8]*, [128 x i8]*, [11 x i8]*) #7 {
entry:
  %3 = alloca [2 x [128 x i8]]
  call void @copy_out([2 x [128 x i8]]* %3, [128 x i8]* %0, [128 x i8]* %1, [11 x i8]* null, [11 x i8]* %2)
  %4 = bitcast [2 x [128 x i8]]* %3 to [128 x i8]*
  %5 = bitcast [11 x i8]* %2 to i8*
  call void @awn_forward_hw_stub([128 x i8]* %4, i8* %5)
  call void @copy_in([2 x [128 x i8]]* %3, [128 x i8]* %0, [128 x i8]* %1, [11 x i8]* null, [11 x i8]* %2)
  ret void
}

declare void @awn_forward_hw_stub([128 x i8]*, i8*)

attributes #0 = { inaccessiblememonly nounwind willreturn }
attributes #1 = { noinline willreturn "fpga.wrapper.func"="wrapper" }
attributes #2 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="arraycpy_hls" }
attributes #3 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #4 = { nounwind willreturn }
attributes #5 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyin" }
attributes #6 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyout" }
attributes #7 = { "fpga.wrapper.func"="stub" }
attributes #8 = { inaccessiblememonly nounwind willreturn "xlx.source"="infer-from-pragma" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}
!datalayout.transforms.on.top = !{!5}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
!5 = !{!6, !8, !10}
!6 = !{!7}
!7 = !{!"0", [2 x [128 x i8]]* null}
!8 = !{!9}
!9 = !{!"array_partition", !"type=Complete", !"dim=1"}
!10 = !{!11, !12}
!11 = !{!"0.0", [128 x i8]* null}
!12 = !{!"0.1", [128 x i8]* null}
!13 = !DILocation(line: 203, column: 5, scope: !14)
!14 = distinct !DISubprogram(name: "awn_forward", linkageName: "_Z11awn_forwardPA128_KaPa", scope: !15, file: !15, line: 200, type: !16, isLocal: false, isDefinition: true, scopeLine: 201, flags: DIFlagPrototyped, isOptimized: false, unit: !29, variables: !4)
!15 = !DIFile(filename: "awn_int8.cpp", directory: "/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/src")
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18, !28}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 1024, elements: !26)
!20 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!21 = !DIDerivedType(tag: DW_TAG_typedef, name: "int8_t", file: !22, line: 24, baseType: !23)
!22 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/stdint-intn.h", directory: "/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/src")
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int8_t", file: !24, line: 37, baseType: !25)
!24 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types.h", directory: "/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/src")
!25 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!26 = !{!27}
!27 = !DISubrange(count: 128)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!29 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !30, producer: "clang version 7.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !31, globals: !40)
!30 = !DIFile(filename: "/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/src/proj_awn_int8/solution1/.autopilot/db/awn_int8.pp.0.cpp", directory: "/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/src")
!31 = !{!32, !35, !21, !37, !34}
!32 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !22, line: 26, baseType: !33)
!33 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int32_t", file: !24, line: 41, baseType: !34)
!34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !36, size: 64)
!36 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !32)
!37 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", file: !22, line: 27, baseType: !38)
!38 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int64_t", file: !24, line: 44, baseType: !39)
!39 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!40 = !{!41, !50, !55, !60, !62, !67, !69, !71, !73, !79, !81, !83, !85, !87, !89, !94, !98, !100, !105, !109, !114}
!41 = !DIGlobalVariableExpression(var: !42, expr: !DIExpression())
!42 = distinct !DIGlobalVariable(name: "W1", linkageName: "_ZL2W1", scope: !29, file: !43, line: 7, type: !44, isLocal: true, isDefinition: true)
!43 = !DIFile(filename: "../golden/awn_weights.h", directory: "/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/src")
!44 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 7168, elements: !45)
!45 = !{!46, !47, !48, !49}
!46 = !DISubrange(count: 64)
!47 = !DISubrange(count: 1)
!48 = !DISubrange(count: 2)
!49 = !DISubrange(count: 7)
!50 = !DIGlobalVariableExpression(var: !51, expr: !DIExpression())
!51 = distinct !DIGlobalVariable(name: "b1", linkageName: "_ZL2b1", scope: !29, file: !52, line: 7, type: !53, isLocal: true, isDefinition: true)
!52 = !DIFile(filename: "../golden/awn_biases.h", directory: "/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/src")
!53 = !DICompositeType(tag: DW_TAG_array_type, baseType: !36, size: 2048, elements: !54)
!54 = !{!46}
!55 = !DIGlobalVariableExpression(var: !56, expr: !DIExpression())
!56 = distinct !DIGlobalVariable(name: "W2", linkageName: "_ZL2W2", scope: !29, file: !43, line: 66, type: !57, isLocal: true, isDefinition: true)
!57 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 163840, elements: !58)
!58 = !{!46, !46, !59}
!59 = !DISubrange(count: 5)
!60 = !DIGlobalVariableExpression(var: !61, expr: !DIExpression())
!61 = distinct !DIGlobalVariable(name: "b2", linkageName: "_ZL2b2", scope: !29, file: !52, line: 14, type: !53, isLocal: true, isDefinition: true)
!62 = !DIGlobalVariableExpression(var: !63, expr: !DIExpression())
!63 = distinct !DIGlobalVariable(name: "Wu1", linkageName: "_ZL3Wu1", scope: !29, file: !43, line: 1349, type: !64, isLocal: true, isDefinition: true)
!64 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 98304, elements: !65)
!65 = !{!46, !46, !66}
!66 = !DISubrange(count: 3)
!67 = !DIGlobalVariableExpression(var: !68, expr: !DIExpression())
!68 = distinct !DIGlobalVariable(name: "Wu4", linkageName: "_ZL3Wu4", scope: !29, file: !43, line: 2120, type: !64, isLocal: true, isDefinition: true)
!69 = !DIGlobalVariableExpression(var: !70, expr: !DIExpression())
!70 = distinct !DIGlobalVariable(name: "bu1", linkageName: "_ZL3bu1", scope: !29, file: !52, line: 21, type: !53, isLocal: true, isDefinition: true)
!71 = !DIGlobalVariableExpression(var: !72, expr: !DIExpression())
!72 = distinct !DIGlobalVariable(name: "bu4", linkageName: "_ZL3bu4", scope: !29, file: !52, line: 28, type: !53, isLocal: true, isDefinition: true)
!73 = !DIGlobalVariableExpression(var: !74, expr: !DIExpression())
!74 = distinct !DIGlobalVariable(name: "TANH_U_LUT", linkageName: "_ZL10TANH_U_LUT", scope: !29, file: !75, line: 76, type: !76, isLocal: true, isDefinition: true)
!75 = !DIFile(filename: "../golden/awn_qparams.h", directory: "/home/nigel/opensource/adversarial-rf/awn_fpga/awn_hls/src")
!76 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 2048, elements: !77)
!77 = !{!78}
!78 = !DISubrange(count: 256)
!79 = !DIGlobalVariableExpression(var: !80, expr: !DIExpression())
!80 = distinct !DIGlobalVariable(name: "Wp1", linkageName: "_ZL3Wp1", scope: !29, file: !43, line: 2891, type: !64, isLocal: true, isDefinition: true)
!81 = !DIGlobalVariableExpression(var: !82, expr: !DIExpression())
!82 = distinct !DIGlobalVariable(name: "Wp4", linkageName: "_ZL3Wp4", scope: !29, file: !43, line: 3662, type: !64, isLocal: true, isDefinition: true)
!83 = !DIGlobalVariableExpression(var: !84, expr: !DIExpression())
!84 = distinct !DIGlobalVariable(name: "bp1", linkageName: "_ZL3bp1", scope: !29, file: !52, line: 35, type: !53, isLocal: true, isDefinition: true)
!85 = !DIGlobalVariableExpression(var: !86, expr: !DIExpression())
!86 = distinct !DIGlobalVariable(name: "bp4", linkageName: "_ZL3bp4", scope: !29, file: !52, line: 42, type: !53, isLocal: true, isDefinition: true)
!87 = !DIGlobalVariableExpression(var: !88, expr: !DIExpression())
!88 = distinct !DIGlobalVariable(name: "TANH_P_LUT", linkageName: "_ZL10TANH_P_LUT", scope: !29, file: !75, line: 94, type: !76, isLocal: true, isDefinition: true)
!89 = !DIGlobalVariableExpression(var: !90, expr: !DIExpression())
!90 = distinct !DIGlobalVariable(name: "Wse0", linkageName: "_ZL4Wse0", scope: !29, file: !43, line: 4433, type: !91, isLocal: true, isDefinition: true)
!91 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 32768, elements: !92)
!92 = !{!93, !27}
!93 = !DISubrange(count: 32)
!94 = !DIGlobalVariableExpression(var: !95, expr: !DIExpression())
!95 = distinct !DIGlobalVariable(name: "Wse3", linkageName: "_ZL4Wse3", scope: !29, file: !43, line: 4692, type: !96, isLocal: true, isDefinition: true)
!96 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 32768, elements: !97)
!97 = !{!27, !93}
!98 = !DIGlobalVariableExpression(var: !99, expr: !DIExpression())
!99 = distinct !DIGlobalVariable(name: "SIGMOID_LUT", linkageName: "_ZL11SIGMOID_LUT", scope: !29, file: !75, line: 112, type: !76, isLocal: true, isDefinition: true)
!100 = !DIGlobalVariableExpression(var: !101, expr: !DIExpression())
!101 = distinct !DIGlobalVariable(name: "Wfc0", linkageName: "_ZL4Wfc0", scope: !29, file: !43, line: 4951, type: !102, isLocal: true, isDefinition: true)
!102 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 327680, elements: !103)
!103 = !{!104, !27}
!104 = !DISubrange(count: 320)
!105 = !DIGlobalVariableExpression(var: !106, expr: !DIExpression())
!106 = distinct !DIGlobalVariable(name: "bfc0", linkageName: "_ZL4bfc0", scope: !29, file: !52, line: 49, type: !107, isLocal: true, isDefinition: true)
!107 = !DICompositeType(tag: DW_TAG_array_type, baseType: !36, size: 10240, elements: !108)
!108 = !{!104}
!109 = !DIGlobalVariableExpression(var: !110, expr: !DIExpression())
!110 = distinct !DIGlobalVariable(name: "Wfc2", linkageName: "_ZL4Wfc2", scope: !29, file: !43, line: 7514, type: !111, isLocal: true, isDefinition: true)
!111 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 28160, elements: !112)
!112 = !{!113, !104}
!113 = !DISubrange(count: 11)
!114 = !DIGlobalVariableExpression(var: !115, expr: !DIExpression())
!115 = distinct !DIGlobalVariable(name: "bfc2", linkageName: "_ZL4bfc2", scope: !29, file: !52, line: 72, type: !116, isLocal: true, isDefinition: true)
!116 = !DICompositeType(tag: DW_TAG_array_type, baseType: !36, size: 352, elements: !117)
!117 = !{!113}
