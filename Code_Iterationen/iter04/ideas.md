# Strong stand need for brupee start
1. Upper Leg (right_upper_leg)

- Position: pos="0.03 -0.2 0" places it slightly to the right and below the torso.
- Joint: right_hip (hinge, axis 0 1 0) allows forward/backward rotation (e.g., for squatting).

    - Range: -120 60:

        -120 lets the leg swing backward (for planking).
        60 lets the leg swing forward (for standing).


- Geometry: A cylinder (size="0.03 0.2") for the thigh.
2. Lower Leg (right_lower_leg)

- Position: pos="0 0 -0.2" extends it downward from the upper leg.
- Joint: right_knee (hinge, axis 0 1 0) allows bending:

    - Range: 0 150:

        0 = fully extended (straight leg).
        150 = deeply bent (for squatting or jumping).


- Geometry: Another cylinder for the shin.


| Motion | Hip Joint Action | Knee Joint Action |
| --- | --- | --- |
| Squat | Rotate forward (~60°) | Bend (~120-150°) |
| Plank | Rotate backward (~-120°) | Fully extended (0°) |
| Jump | Rotate forward (~60°) | Bend (~120°), then extend (0°) |
