# Mutation re-check of the 6 survivors (temporary copy outside the repo; mutants applied only there)

| mutant | status | killed by |
|---|---|---|
| M1c CLI/server default scope -> all | KILLED | tests/test_review_s_fixes.py::test_server_query_defaults_to_the_evaluated_scope |
| M2a loop compression ratio 2.4 -> 9 | KILLED | tests/test_review_s_fixes.py::test_chunk_looped_zlib_branch_has_its_threshold_at_2_4 |
| M4a rep_ctx_p1 route_info experimental flag removed | KILLED | tests/test_review_s_fixes.py::test_rep_ctx_p1_route_info_is_marked_experimental_and_not_validated_outside_dihana |
| M4b text-route event experimental flag removed | KILLED | tests/test_review_s_fixes.py::test_rep_ctx_p1_events_are_nonempty_experimental_and_say_so |
| M4c route_info validated_outside_dihana -> True | KILLED | tests/test_review_s_fixes.py::test_rep_ctx_p1_route_info_is_marked_experimental_and_not_validated_outside_dihana |
| X1 offset not recorded | KILLED | tests/test_review_s_fixes.py::test_redecode_events_record_the_chunk_offset_and_reproducibility |
| X2 reproducible always True | KILLED | tests/test_review_s_fixes.py::test_redecode_events_record_the_chunk_offset_and_reproducibility |
| X3 device warning always on | KILLED | tests/test_review_s_fixes.py::test_p0_on_gpu_through_auto_carries_a_device_warning_and_cpu_does_not |
| X4 noloop warning dropped | KILLED | tests/test_review_s_fixes.py::test_service_accepts_the_opt_in_profiles_as_experimental_with_a_specific_warning[p0_noloop] |
| X5 device error uses cache message | KILLED | tests/test_review_s_fixes.py::test_cuda_profile_without_a_device_is_a_structured_error_that_says_so |
| M8 unresolved keeps re-decoded | KILLED | tests/test_review_s_fixes.py::test_unresolved_chunk_keeps_the_original_text_not_the_last_redecode |

M1c, M2a, M4a, M4b, M4c, M8 = the six survivors of REVISION_S; X1-X5 = extra mutants of the new behaviour (offset, reproducible flag, auto-GPU warning, noloop warning, device error message). Sandbox note: the temporary copy and the review clones were not deleted.
