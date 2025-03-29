import numpy as np

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]

smpl_raw_offsets = np.array([[0,0,0],
                           [0.5600,-0.8082,-0.1820],
                           [-0.5362,-0.8335,-0.1333],
                           [0.0371, 0.9555, -0.2926],
                           [0.1169,-0.9928,0.0265],
                           [-0.1163,-0.9932,-0.0080],
                           [0.0309, 0.9821, 0.1858],
                           [-0.0312,-0.9963,-0.0797],
                           [0.0424,-0.9963,-0.0749],
                           [-0.0352,0.9992,0.0161],
                           [0.3203,-0.4288,0.8447],
                           [-0.2352,-0.4289,0.8722],
                           [-0.0635,0.9902,-0.1247],
                           [0.5334,0.8349,-0.1358],
                           [-0.5928,0.7890,-0.1613],
                           [0.1024,0.8624,0.4958],
                           [0.9239,0.3577,-0.1360],
                           [-0.9158,0.3967,-0.0622],
                           [0.9941,-0.0656,-0.0867],
                           [-0.9919,-0.0636,-0.1100],
                           [0.9983,0.0486,-0.0309],
                           [-0.9992,0.0329,-0.0223]])


smpl_real_offsets = np.array([[ 0.0000,  0.0000,  0.0000],
        [ 0.0577, -0.0833, -0.0188],
        [-0.0589, -0.0916, -0.0146],
        [ 0.0049,  0.1257, -0.0385],
        [ 0.0460, -0.3907,  0.0104],
        [-0.0454, -0.3874, -0.0031],
        [ 0.0044,  0.1406,  0.0266],
        [-0.0135, -0.4304, -0.0344],
        [ 0.0180, -0.4241, -0.0319],
        [-0.0020,  0.0574,  0.0009],
        [ 0.0459, -0.0614,  0.1210],
        [-0.0351, -0.0641,  0.1303],
        [-0.0139,  0.2172, -0.0274],
        [ 0.0733,  0.1147, -0.0187],
        [-0.0850,  0.1131, -0.0231],
        [ 0.0105,  0.0888,  0.0511],
        [ 0.1215,  0.0470, -0.0179],
        [-0.1126,  0.0488, -0.0076],
        [ 0.2555, -0.0169, -0.0223],
        [-0.2610, -0.0167, -0.0289],
        [ 0.2657,  0.0129, -0.0082],
        [-0.2696,  0.0089, -0.0060]])


bandai_kinematic_chain = [[0, 17, 18, 19, 20], [0, 13, 14, 15, 16], [0, 1, 2, 3, 4], [2, 9, 10, 11, 12], [2, 5, 6, 7, 8]]
bandai_raw_offsets = np.array([[ 0.        ,  0.          ,0. ,       ],
                               [ 0.03393824,  0.9968272  ,-0.07199915],
                               [ 0.03112892 , 0.9983263  ,-0.04873997],
                               [ 0.02526809  ,0.9996783  ,-0.00217649],
                               [ 0.01548805 , 0.9976811  , 0.0662759 ],
                               [ 0.3816068  , 0.9242508  ,-0.01169344],
                               [ 0.9547773  , 0.07825354 ,-0.28683922],
                               [ 0.9968188  ,-0.03057985 ,0.07360202],
                               [ 0.998804   ,-0.04302648 ,0.02322503],
                               [-0.3336623  , 0.94229907 ,-0.02723938],
                               [-0.935373   ,-0.01458865 ,-0.3533616 ],
                               [-0.9984354  ,-0.03674256  ,0.04215111],
                               [-0.9998896  ,-0.01312141  ,0.00697517],
                               [ 0.9355912  ,-0.35123464  ,0.03609942],
                               [ 0.04325764 ,-0.993617    ,0.10418281],
                               [-0.00126767 ,-0.95700693 ,-0.2900625 ],
                               [ 0.04622368 ,-0.6362332  , 0.7701108 ],
                               [-0.9573763  ,-0.2886664   ,0.01011907],
                               [-0.0505516  ,-0.996677    ,0.0638713 ],
                               [ 0.03451919 ,-0.96347344 ,-0.2655699 ],
                               [-0.1024961  ,-0.6355697   ,0.76520956]])

bandai_real_offsets = np.array([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 5.4407e-03,  1.5980e-01, -1.1542e-02],
        [ 2.8522e-03,  9.1473e-02, -4.4658e-03],
        [ 4.2159e-03,  1.6679e-01, -3.6314e-04],
        [ 1.0188e-03,  6.5627e-02,  4.3596e-03],
        [ 5.4109e-02,  1.3105e-01, -1.6581e-03],
        [ 5.2021e-02,  4.2636e-03, -1.5628e-02],
        [ 2.4707e-01, -7.5793e-03,  1.8243e-02],
        [ 2.2683e-01, -9.7714e-03,  5.2744e-03],
        [-4.7312e-02,  1.3362e-01, -3.8625e-03],
        [-5.0964e-02, -7.9487e-04, -1.9253e-02],
        [-2.4747e-01, -9.1069e-03,  1.0447e-02],
        [-2.2708e-01, -2.9799e-03,  1.5841e-03],
        [ 6.5109e-02, -2.4443e-02,  2.5122e-03],
        [ 1.7223e-02, -3.9561e-01,  4.1480e-02],
        [-5.2932e-04, -3.9960e-01, -1.2112e-01],
        [ 5.5353e-03, -7.6189e-02,  9.2221e-02],
        [-6.6625e-02, -2.0089e-02,  7.0420e-04],
        [-2.0127e-02, -3.9683e-01,  2.5430e-02],
        [ 1.4413e-02, -4.0230e-01, -1.1089e-01],
        [-1.2274e-02, -7.6110e-02,  9.1634e-02]])

kit_tgt_skel_id = '03950'

t2m_tgt_skel_id = '000021'

xia_kinematic_chain = [[0, 1, 12, 13, 14, 15], [0, 1, 16, 17, 18, 19], [0, 1, 2, 11], [2, 3, 4, 5, 6], [2, 7, 8, 9, 10]]
xia_raw_offsets = np.array([[ 0.,          0.,          0.        ],
                            [ 0.,          0.,         0.        ],
                            [ 0.,          0.9773194,  -0.21177018],
                            [ 0.,          1.,          0.        ],
                            [-0.95751137,  0.28839538,  0.        ],
                            [-1.,          0.,          0.        ],
                            [-1.,          0.,         0.        ],
                            [ 0.,          1.,          0.        ],
                            [ 0.95751137,  0.28839538,  0.        ],
                            [ 1.,          0.,          0.        ],
                            [ 1.,          0.,          0.        ],
                            [ 0.,          1.,          0.        ],
                            [-1.,          0.,          0.        ],
                            [ 0.,         -1.,          0.        ],
                            [ 0.,         -1.,          0.        ],
                            [ 0.,          0.,          1.        ],
                            [ 1.,          0.,          0.        ],
                            [ 0.,         -1.,          0.        ],
                            [ 0.,         -1.,         0.        ],
                            [ 0.,          0.,          1.        ]])

xia_real_offsets = np.array([[ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.2305, -0.0499],
        [ 0.0000,  0.2797,  0.0000],
        [-0.1855,  0.0559,  0.0000],
        [-0.2482,  0.0000,  0.0000],
        [-0.2452,  0.0000,  0.0000],
        [ 0.0000,  0.2797,  0.0000],
        [ 0.1855,  0.0559,  0.0000],
        [ 0.2482,  0.0000,  0.0000],
        [ 0.2452,  0.0000,  0.0000],
        [ 0.0000,  0.3517,  0.0000],
        [-0.0827,  0.0000,  0.0000],
        [ 0.0000, -0.4332,  0.0000],
        [ 0.0000, -0.3825,  0.0000],
        [ 0.0000,  0.0000,  0.1659],
        [ 0.0827,  0.0000,  0.0000],
        [ 0.0000, -0.4332,  0.0000],
        [ 0.0000, -0.3825,  0.0000],
        [ 0.0000,  0.0000,  0.1659]], dtype=np.float32)