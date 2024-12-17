function F = computeF(intrinsics,first_pose,second_pose)
rot_1 = first_pose.R;
trans_1 = first_pose.Translation';

rot_2 = second_pose.R;
trans_2 = second_pose.Translation';

rot_1to2 = rot_1'*rot_2;
trans_1to2 = rot_1'*(trans_2-trans_1);

trans_1to2_skew = [0, -trans_1to2(3), trans_1to2(2); trans_1to2(3), 0, -trans_1to2(1);...
    -trans_1to2(2) trans_1to2(1) 0];

F = intrinsics.K' \ trans_1to2_skew * rot_1to2 / intrinsics.K;

end
