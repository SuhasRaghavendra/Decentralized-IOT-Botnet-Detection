window.DASHBOARD_DATA = {
  "generated_at": "2026-06-20T17:06:09.337287+00:00",
  "overview": {
    "project_title": "Decentralized IoT Botnet Detection",
    "dataset": "CICIoT2023",
    "targeted_attacks": [
      {
        "key": "ddos_icmp",
        "label": "DDoS-ICMP Flood",
        "color": "#e05c5c",
        "pct": 15.3
      },
      {
        "key": "ddos_syn",
        "label": "DDoS-SYN Flood",
        "color": "#4f9de8",
        "pct": 8.7
      },
      {
        "key": "mirai_greeth",
        "label": "Mirai-Greeth_flood",
        "color": "#48c48e",
        "pct": 5.0
      }
    ],
    "stats": {
      "total_samples": 7845673,
      "train_samples": 5491971,
      "val_samples": 1176851,
      "test_samples": 1176851,
      "selected_features": 17,
      "attack_types": 34,
      "family_classes": 15,
      "models_trained": 16,
      "best_binary_f1": 0.9966812666556971,
      "fl_rounds": 2,
      "fl_clients": 2,
      "graph_nodes": 20,
      "graph_edges": 190
    },
    "pipeline_stages": [
      {
        "id": "preprocess",
        "label": "Preprocessing",
        "status": "done",
        "detail": "Clean + scale 7.8M rows, 17 features selected"
      },
      {
        "id": "baseline",
        "label": "Baseline Models",
        "status": "done",
        "detail": "LR / RF / XGB / LGBM \u2014 best binary F1 = 0.9967"
      },
      {
        "id": "graph",
        "label": "Graph Construction",
        "status": "done",
        "detail": "20 nodes, 190 edges, 4 partitions"
      },
      {
        "id": "spectral",
        "label": "Spectral Analysis",
        "status": "done",
        "detail": "Normalized Laplacian, top-8 eigenvalues, Fiedler value"
      },
      {
        "id": "attack",
        "label": "Attack-Specific ML",
        "status": "done",
        "detail": "Per-attack OVR models for 3 targeted attacks"
      },
      {
        "id": "federated",
        "label": "Federated Learning",
        "status": "done",
        "detail": "Flower FL, 2 clients, 2 rounds, global F1 = 0.9964583442106546"
      }
    ]
  },
  "preprocessing": {
    "run_timestamp": "2026-04-28T12:27:53Z",
    "total_rows": {
      "train": 5491971,
      "validation": 1176851,
      "test": 1176851
    },
    "feature_count": 17,
    "selected_features": [
      "Header_Length",
      "Duration",
      "syn_flag_number",
      "ack_flag_number",
      "syn_count",
      "urg_count",
      "rst_count",
      "HTTPS",
      "UDP",
      "ICMP",
      "Tot sum",
      "Min",
      "Max",
      "AVG",
      "Tot size",
      "Covariance",
      "Variance"
    ],
    "dropped_constant": [
      "Telnet",
      "IRC"
    ],
    "dropped_correlated": [
      "Srate",
      "rst_flag_number",
      "ack_count",
      "LLC",
      "Std",
      "Number",
      "Magnitue",
      "Radius",
      "Weight"
    ],
    "correlation_threshold": 0.95,
    "selection_method": "binary_pearson_top20_intersect_binary_mi_top20",
    "feature_table": [
      {
        "feature": "Protocol Type",
        "pearson_r": 0.0,
        "mutual_info": 0.223339,
        "selected": false
      },
      {
        "feature": "Min",
        "pearson_r": 0.098982,
        "mutual_info": 0.19949,
        "selected": true
      },
      {
        "feature": "Tot size",
        "pearson_r": 0.31801,
        "mutual_info": 0.125086,
        "selected": true
      },
      {
        "feature": "AVG",
        "pearson_r": 0.319333,
        "mutual_info": 0.113702,
        "selected": true
      },
      {
        "feature": "Tot sum",
        "pearson_r": 0.309034,
        "mutual_info": 0.111941,
        "selected": true
      },
      {
        "feature": "Max",
        "pearson_r": 0.454556,
        "mutual_info": 0.101563,
        "selected": true
      },
      {
        "feature": "Duration",
        "pearson_r": 0.540432,
        "mutual_info": 0.101337,
        "selected": true
      },
      {
        "feature": "IAT",
        "pearson_r": 0.0,
        "mutual_info": 0.097543,
        "selected": false
      },
      {
        "feature": "urg_count",
        "pearson_r": 0.242577,
        "mutual_info": 0.086623,
        "selected": true
      },
      {
        "feature": "rst_count",
        "pearson_r": 0.493519,
        "mutual_info": 0.08605,
        "selected": true
      },
      {
        "feature": "Variance",
        "pearson_r": 0.510377,
        "mutual_info": 0.082661,
        "selected": true
      },
      {
        "feature": "flow_duration",
        "pearson_r": 0.0,
        "mutual_info": 0.067687,
        "selected": false
      },
      {
        "feature": "ack_flag_number",
        "pearson_r": 0.342874,
        "mutual_info": 0.065193,
        "selected": true
      },
      {
        "feature": "Covariance",
        "pearson_r": 0.295815,
        "mutual_info": 0.060154,
        "selected": true
      },
      {
        "feature": "Header_Length",
        "pearson_r": 0.316222,
        "mutual_info": 0.057628,
        "selected": true
      },
      {
        "feature": "syn_count",
        "pearson_r": 0.110435,
        "mutual_info": 0.053057,
        "selected": true
      },
      {
        "feature": "syn_flag_number",
        "pearson_r": 0.079418,
        "mutual_info": 0.051284,
        "selected": true
      },
      {
        "feature": "ICMP",
        "pearson_r": 0.068805,
        "mutual_info": 0.049603,
        "selected": true
      },
      {
        "feature": "HTTPS",
        "pearson_r": 0.446243,
        "mutual_info": 0.049204,
        "selected": true
      },
      {
        "feature": "UDP",
        "pearson_r": 0.054805,
        "mutual_info": 0.047224,
        "selected": true
      },
      {
        "feature": "TCP",
        "pearson_r": 0.090364,
        "mutual_info": 0.0,
        "selected": false
      },
      {
        "feature": "fin_count",
        "pearson_r": 0.042265,
        "mutual_info": 0.0,
        "selected": false
      },
      {
        "feature": "fin_flag_number",
        "pearson_r": 0.047823,
        "mutual_info": 0.0,
        "selected": false
      }
    ],
    "class_distribution": [
      {
        "family": "Backdoor",
        "weight": 934.0086734693878,
        "share": 0.0010706538690754195,
        "pct": 0.01
      },
      {
        "family": "Benign",
        "weight": 2.826440117957665,
        "share": 0.3538019410517645,
        "pct": 2.36
      },
      {
        "family": "BrowserHijacking",
        "weight": 550.5735338345864,
        "share": 0.0018162878136100866,
        "pct": 0.01
      },
      {
        "family": "CommandInjection",
        "weight": 590.5345161290322,
        "share": 0.0016933811194560205,
        "pct": 0.01
      },
      {
        "family": "DDoS",
        "weight": 0.09156718769538577,
        "share": 10.920942590556287,
        "pct": 72.81
      },
      {
        "family": "DNS",
        "weight": 17.258951635712265,
        "share": 0.0579409468840968,
        "pct": 0.39
      },
      {
        "family": "DictionaryBruteForce",
        "weight": 237.59338092147956,
        "share": 0.004208871459809238,
        "pct": 0.03
      },
      {
        "family": "DoS",
        "weight": 0.38513552746734886,
        "share": 2.596488583060617,
        "pct": 17.31
      },
      {
        "family": "MITM",
        "weight": 10.081820685097478,
        "share": 0.09918843344220135,
        "pct": 0.66
      },
      {
        "family": "Mirai",
        "weight": 1.1819535910746106,
        "share": 0.8460569074381493,
        "pct": 5.64
      },
      {
        "family": "Recon",
        "weight": 9.836688965906344,
        "share": 0.10166022362463312,
        "pct": 0.68
      },
      {
        "family": "SqlInjection",
        "weight": 620.5616949152543,
        "share": 0.0016114433233533096,
        "pct": 0.01
      },
      {
        "family": "Uploading",
        "weight": 2615.2242857142855,
        "share": 0.0003823763818126498,
        "pct": 0.0
      },
      {
        "family": "VulnerabilityScan",
        "weight": 83.28739763421292,
        "share": 0.012006618388917202,
        "pct": 0.08
      },
      {
        "family": "XSS",
        "weight": 884.3753623188405,
        "share": 0.0011307415862174073,
        "pct": 0.01
      }
    ],
    "pipeline_steps": [
      {
        "step": 1,
        "name": "Load Raw CSVs",
        "detail": "Train: 5.5M rows, Val: 1.18M, Test: 1.18M"
      },
      {
        "step": 2,
        "name": "Create Targets",
        "detail": "label_binary (0/1) + label_family (15 classes)"
      },
      {
        "step": 3,
        "name": "Clean Numeric",
        "detail": "inf \u2192 train max/min, NaN \u2192 train median (no leakage)"
      },
      {
        "step": 4,
        "name": "Drop Constant Columns",
        "detail": "Removed: ['Telnet', 'IRC']"
      },
      {
        "step": 5,
        "name": "Drop Correlated Columns",
        "detail": "Removed 9 columns at r>0.95"
      },
      {
        "step": 6,
        "name": "Feature Ranking",
        "detail": "Pearson correlation + Mutual Information (top-20 each)"
      },
      {
        "step": 7,
        "name": "Feature Selection",
        "detail": "Intersection: 17 features selected"
      },
      {
        "step": 8,
        "name": "StandardScaler",
        "detail": "Fit on train only; transform all splits"
      }
    ]
  },
  "baseline": {
    "binary_results": [
      {
        "model": "RF",
        "accuracy": 0.9935216947599994,
        "precision": 0.9972899044366816,
        "recall": 0.9960733713148159,
        "f1": 0.9966812666556971
      },
      {
        "model": "LGBM",
        "accuracy": 0.9932820722419405,
        "precision": 0.9972909717647633,
        "recall": 0.9958262712601754,
        "f1": 0.9965580833234653
      },
      {
        "model": "XGB",
        "accuracy": 0.9911050761736193,
        "precision": 0.9951539551829984,
        "recall": 0.9957410043399122,
        "f1": 0.9954473932107534
      },
      {
        "model": "LR",
        "accuracy": 0.9872558208303345,
        "precision": 0.9921965985083978,
        "recall": 0.9947743558867238,
        "f1": 0.9934838050963439
      }
    ],
    "family_results": [
      {
        "model": "RF",
        "accuracy": 0.8867256772522605,
        "precision": null,
        "recall": null,
        "f1": 0.45590109645681587
      },
      {
        "model": "XGB",
        "accuracy": 0.8849489017726119,
        "precision": null,
        "recall": null,
        "f1": 0.45490321452003957
      },
      {
        "model": "LGBM",
        "accuracy": 0.8514629294617585,
        "precision": null,
        "recall": null,
        "f1": 0.3414812605893
      },
      {
        "model": "LR",
        "accuracy": 0.8267231790600509,
        "precision": null,
        "recall": null,
        "f1": 0.27034177099301004
      }
    ],
    "best_binary_model": "RF",
    "best_binary_f1": 0.9966812666556971
  },
  "attacks": {
    "attacks": {
      "ddos_icmp": {
        "attack": "ddos_icmp",
        "attack_display": "DDoS-ICMP Flood",
        "best_model": "rf",
        "features": [
          "Header_Length",
          "Duration",
          "syn_flag_number",
          "ack_flag_number",
          "syn_count",
          "urg_count",
          "rst_count",
          "HTTPS",
          "UDP",
          "ICMP",
          "Tot sum",
          "Min",
          "Max",
          "AVG",
          "Tot size",
          "Covariance",
          "Variance",
          "Protocol Type",
          "Rate",
          "flow_duration",
          "Number",
          "TCP"
        ],
        "extra_features": [
          "Protocol Type",
          "Rate",
          "flow_duration",
          "Number",
          "TCP"
        ],
        "rows": {
          "train_total": 5491971,
          "train_attack": 848088,
          "train_benign": 4643883,
          "validation": 1176851,
          "test": 1176851
        },
        "results": [
          {
            "model": "lr",
            "accuracy": 0.9994077415067838,
            "precision": 0.9973447734828452,
            "recall": 0.9988297410596063,
            "f1": 0.998086704932568,
            "roc_auc": 0.9996256947669314,
            "train_time_s": 10.77
          },
          {
            "model": "rf",
            "accuracy": 0.9998402516546274,
            "precision": 0.9999945002062423,
            "recall": 0.9989725895687623,
            "f1": 0.9994832836772611,
            "roc_auc": 0.9999936720275993,
            "train_time_s": 158.5
          },
          {
            "model": "xgb",
            "accuracy": 0.9997901178653882,
            "precision": 0.9994229944937189,
            "recall": 0.9992198273730709,
            "f1": 0.9993214006071679,
            "roc_auc": 0.9999992724157871,
            "train_time_s": 23.14
          },
          {
            "model": "lgbm",
            "accuracy": 0.9623928602686321,
            "precision": 0.815549192532356,
            "recall": 0.9780397888039734,
            "f1": 0.8894340546509246,
            "roc_auc": 0.9688566785069284,
            "train_time_s": 32.77
          }
        ],
        "best": {
          "model": "rf",
          "accuracy": 0.9998402516546274,
          "precision": 0.9999945002062423,
          "recall": 0.9989725895687623,
          "f1": 0.9994832836772611,
          "roc_auc": 0.9999936720275993,
          "train_time_s": 158.5
        },
        "confusion_matrix": [
          [
            994839,
            1
          ],
          [
            187,
            181824
          ]
        ],
        "classification_report": {
          "Benign": {
            "precision": 0.9998120652123663,
            "recall": 0.9999989948132363,
            "f1-score": 0.999905521276307,
            "support": 994840.0
          },
          "Attack": {
            "precision": 0.9999945002062423,
            "recall": 0.9989725895687623,
            "f1-score": 0.9994832836772611,
            "support": 182011.0
          },
          "accuracy": 0.9998402516546274,
          "macro avg": {
            "precision": 0.9999032827093043,
            "recall": 0.9994857921909993,
            "f1-score": 0.9996944024767841,
            "support": 1176851.0
          },
          "weighted avg": {
            "precision": 0.9998402804882767,
            "recall": 0.9998402516546274,
            "f1-score": 0.9998402182875346,
            "support": 1176851.0
          }
        },
        "feature_importance": {
          "lr": [
            {
              "feature": "ICMP",
              "importance": 3.999252
            },
            {
              "feature": "TCP",
              "importance": 2.929044
            },
            {
              "feature": "Tot sum",
              "importance": 1.353956
            },
            {
              "feature": "Tot size",
              "importance": 1.235986
            },
            {
              "feature": "AVG",
              "importance": 1.138785
            },
            {
              "feature": "ack_flag_number",
              "importance": 1.138444
            },
            {
              "feature": "Protocol Type",
              "importance": 1.013641
            },
            {
              "feature": "syn_flag_number",
              "importance": 0.947266
            },
            {
              "feature": "Covariance",
              "importance": 0.815498
            },
            {
              "feature": "Max",
              "importance": 0.683832
            }
          ],
          "rf": [
            {
              "feature": "Min",
              "importance": 0.228759
            },
            {
              "feature": "AVG",
              "importance": 0.178949
            },
            {
              "feature": "Tot size",
              "importance": 0.156723
            },
            {
              "feature": "Protocol Type",
              "importance": 0.12087
            },
            {
              "feature": "Tot sum",
              "importance": 0.078472
            },
            {
              "feature": "ICMP",
              "importance": 0.071278
            },
            {
              "feature": "Max",
              "importance": 0.069623
            },
            {
              "feature": "Header_Length",
              "importance": 0.034641
            },
            {
              "feature": "TCP",
              "importance": 0.02597
            },
            {
              "feature": "UDP",
              "importance": 0.010557
            }
          ],
          "xgb": [
            {
              "feature": "Min",
              "importance": 0.973233
            },
            {
              "feature": "ICMP",
              "importance": 0.017095
            },
            {
              "feature": "Variance",
              "importance": 0.001761
            },
            {
              "feature": "rst_count",
              "importance": 0.001718
            },
            {
              "feature": "Number",
              "importance": 0.001211
            },
            {
              "feature": "HTTPS",
              "importance": 0.000983
            },
            {
              "feature": "Protocol Type",
              "importance": 0.000821
            },
            {
              "feature": "UDP",
              "importance": 0.000577
            },
            {
              "feature": "Tot size",
              "importance": 0.000425
            },
            {
              "feature": "TCP",
              "importance": 0.000401
            }
          ],
          "lgbm": [
            {
              "feature": "Rate",
              "importance": 1330
            },
            {
              "feature": "Protocol Type",
              "importance": 1257
            },
            {
              "feature": "flow_duration",
              "importance": 1125
            },
            {
              "feature": "Duration",
              "importance": 961
            },
            {
              "feature": "Min",
              "importance": 926
            },
            {
              "feature": "Header_Length",
              "importance": 843
            },
            {
              "feature": "Variance",
              "importance": 839
            },
            {
              "feature": "Tot size",
              "importance": 669
            },
            {
              "feature": "urg_count",
              "importance": 659
            },
            {
              "feature": "syn_count",
              "importance": 641
            }
          ]
        }
      },
      "ddos_syn": {
        "attack": "ddos_syn",
        "attack_display": "DDoS-SYN Flood",
        "best_model": "rf",
        "features": [
          "Header_Length",
          "Duration",
          "syn_flag_number",
          "ack_flag_number",
          "syn_count",
          "urg_count",
          "rst_count",
          "HTTPS",
          "UDP",
          "ICMP",
          "Tot sum",
          "Min",
          "Max",
          "AVG",
          "Tot size",
          "Covariance",
          "Variance",
          "TCP",
          "Rate",
          "flow_duration",
          "fin_flag_number",
          "psh_flag_number",
          "Number"
        ],
        "extra_features": [
          "TCP",
          "Rate",
          "flow_duration",
          "fin_flag_number",
          "psh_flag_number",
          "Number"
        ],
        "rows": {
          "train_total": 5491971,
          "train_attack": 716226,
          "train_benign": 4775745,
          "validation": 1176851,
          "test": 1176851
        },
        "results": [
          {
            "model": "lr",
            "accuracy": 0.926654266342978,
            "precision": 0.6428105571648794,
            "recall": 0.9871487197497383,
            "f1": 0.778607941356766,
            "roc_auc": 0.978504783870714,
            "train_time_s": 22.2
          },
          {
            "model": "rf",
            "accuracy": 0.9772630519921384,
            "precision": 0.8542230304517842,
            "recall": 0.9959351972892644,
            "f1": 0.9196519190693819,
            "roc_auc": 0.9924947204380721,
            "train_time_s": 190.69
          },
          {
            "model": "xgb",
            "accuracy": 0.9760190542388119,
            "precision": 0.846429379591251,
            "recall": 0.9974180373181407,
            "f1": 0.9157416419363121,
            "roc_auc": 0.9929106180840125,
            "train_time_s": 25.18
          },
          {
            "model": "lgbm",
            "accuracy": 0.9767124300357479,
            "precision": 0.8501809756611293,
            "recall": 0.9975481110048843,
            "f1": 0.9179878504952569,
            "roc_auc": 0.9931620303963508,
            "train_time_s": 30.64
          }
        ],
        "best": {
          "model": "rf",
          "accuracy": 0.9772630519921384,
          "precision": 0.8542230304517842,
          "recall": 0.9959351972892644,
          "f1": 0.9196519190693819,
          "roc_auc": 0.9924947204380721,
          "train_time_s": 190.69
        },
        "confusion_matrix": [
          [
            996959,
            26133
          ],
          [
            625,
            153134
          ]
        ],
        "classification_report": {
          "Benign": {
            "precision": 0.9993734863430047,
            "recall": 0.9744568425908912,
            "f1-score": 0.9867578968622381,
            "support": 1023092.0
          },
          "Attack": {
            "precision": 0.8542230304517842,
            "recall": 0.9959351972892644,
            "f1-score": 0.9196519190693819,
            "support": 153759.0
          },
          "accuracy": 0.9772630519921384,
          "macro avg": {
            "precision": 0.9267982583973944,
            "recall": 0.9851960199400778,
            "f1-score": 0.95320490796581,
            "support": 1176851.0
          },
          "weighted avg": {
            "precision": 0.9804091578533505,
            "recall": 0.9772630519921384,
            "f1-score": 0.9779903060291999,
            "support": 1176851.0
          }
        },
        "feature_importance": {
          "lr": [
            {
              "feature": "syn_flag_number",
              "importance": 3.236042
            },
            {
              "feature": "urg_count",
              "importance": 3.07411
            },
            {
              "feature": "psh_flag_number",
              "importance": 2.324507
            },
            {
              "feature": "ICMP",
              "importance": 1.483052
            },
            {
              "feature": "TCP",
              "importance": 1.332159
            },
            {
              "feature": "syn_count",
              "importance": 1.265863
            },
            {
              "feature": "fin_flag_number",
              "importance": 1.096115
            },
            {
              "feature": "Tot sum",
              "importance": 0.99247
            },
            {
              "feature": "AVG",
              "importance": 0.705375
            },
            {
              "feature": "rst_count",
              "importance": 0.647026
            }
          ],
          "rf": [
            {
              "feature": "syn_flag_number",
              "importance": 0.322121
            },
            {
              "feature": "syn_count",
              "importance": 0.318415
            },
            {
              "feature": "TCP",
              "importance": 0.054948
            },
            {
              "feature": "Tot sum",
              "importance": 0.037091
            },
            {
              "feature": "Min",
              "importance": 0.029367
            },
            {
              "feature": "Tot size",
              "importance": 0.027497
            },
            {
              "feature": "flow_duration",
              "importance": 0.027183
            },
            {
              "feature": "Header_Length",
              "importance": 0.025069
            },
            {
              "feature": "urg_count",
              "importance": 0.024571
            },
            {
              "feature": "Rate",
              "importance": 0.023169
            }
          ],
          "xgb": [
            {
              "feature": "syn_flag_number",
              "importance": 0.953176
            },
            {
              "feature": "syn_count",
              "importance": 0.014327
            },
            {
              "feature": "rst_count",
              "importance": 0.013092
            },
            {
              "feature": "flow_duration",
              "importance": 0.005008
            },
            {
              "feature": "urg_count",
              "importance": 0.003879
            },
            {
              "feature": "Number",
              "importance": 0.001777
            },
            {
              "feature": "ack_flag_number",
              "importance": 0.001507
            },
            {
              "feature": "Rate",
              "importance": 0.001497
            },
            {
              "feature": "AVG",
              "importance": 0.000966
            },
            {
              "feature": "Header_Length",
              "importance": 0.000553
            }
          ],
          "lgbm": [
            {
              "feature": "Rate",
              "importance": 2097
            },
            {
              "feature": "syn_count",
              "importance": 1475
            },
            {
              "feature": "flow_duration",
              "importance": 1377
            },
            {
              "feature": "Header_Length",
              "importance": 1026
            },
            {
              "feature": "rst_count",
              "importance": 988
            },
            {
              "feature": "Duration",
              "importance": 836
            },
            {
              "feature": "Tot size",
              "importance": 606
            },
            {
              "feature": "Variance",
              "importance": 582
            },
            {
              "feature": "urg_count",
              "importance": 581
            },
            {
              "feature": "Min",
              "importance": 529
            }
          ]
        }
      },
      "mirai_greeth": {
        "attack": "mirai_greeth",
        "attack_display": "Mirai-Greeth_flood",
        "best_model": "lgbm",
        "features": [
          "Header_Length",
          "Duration",
          "syn_flag_number",
          "ack_flag_number",
          "syn_count",
          "urg_count",
          "rst_count",
          "HTTPS",
          "UDP",
          "ICMP",
          "Tot sum",
          "Min",
          "Max",
          "AVG",
          "Tot size",
          "Covariance",
          "Variance",
          "Protocol Type",
          "Rate",
          "flow_duration",
          "Number",
          "TCP"
        ],
        "extra_features": [
          "Protocol Type",
          "Rate",
          "flow_duration",
          "Number",
          "TCP"
        ],
        "rows": {
          "train_total": 5491971,
          "train_attack": 116133,
          "train_benign": 5375838,
          "validation": 1176851,
          "test": 1176851
        },
        "results": [
          {
            "model": "lr",
            "accuracy": 0.9837498544845524,
            "precision": 0.5660683292226181,
            "recall": 0.9950622240064231,
            "f1": 0.7216221724067658,
            "roc_auc": 0.9924069056068897,
            "train_time_s": 14.12
          },
          {
            "model": "rf",
            "accuracy": 0.9988350266941185,
            "precision": 0.9521339940839768,
            "recall": 0.9949819349658772,
            "f1": 0.9730865118470388,
            "roc_auc": 0.9999596867037988,
            "train_time_s": 124.13
          },
          {
            "model": "xgb",
            "accuracy": 0.9977252855289243,
            "precision": 0.9036932127682754,
            "recall": 0.9989963869931754,
            "f1": 0.9489579956908879,
            "roc_auc": 0.9999720333616893,
            "train_time_s": 27.98
          },
          {
            "model": "lgbm",
            "accuracy": 0.9989284964706662,
            "precision": 0.9557174239796509,
            "recall": 0.995503813729426,
            "f1": 0.9752049865308611,
            "roc_auc": 0.9999766447988834,
            "train_time_s": 31.4
          }
        ],
        "best": {
          "model": "lgbm",
          "accuracy": 0.9989284964706662,
          "precision": 0.9557174239796509,
          "recall": 0.995503813729426,
          "f1": 0.9752049865308611,
          "roc_auc": 0.9999766447988834,
          "train_time_s": 31.4
        },
        "confusion_matrix": [
          [
            1150792,
            1149
          ],
          [
            112,
            24798
          ]
        ],
        "classification_report": {
          "Benign": {
            "precision": 0.9999026851935522,
            "recall": 0.9990025530821457,
            "f1-score": 0.9994524164674565,
            "support": 1151941.0
          },
          "Attack": {
            "precision": 0.9557174239796509,
            "recall": 0.995503813729426,
            "f1-score": 0.9752049865308611,
            "support": 24910.0
          },
          "accuracy": 0.9989284964706662,
          "macro avg": {
            "precision": 0.9778100545866015,
            "recall": 0.9972531834057858,
            "f1-score": 0.9873287014991587,
            "support": 1176851.0
          },
          "weighted avg": {
            "precision": 0.9989674309796897,
            "recall": 0.9989284964706662,
            "f1-score": 0.9989391794648789,
            "support": 1176851.0
          }
        },
        "feature_importance": {
          "lr": [
            {
              "feature": "Protocol Type",
              "importance": 2.579953
            },
            {
              "feature": "urg_count",
              "importance": 1.187584
            },
            {
              "feature": "Variance",
              "importance": 1.103587
            },
            {
              "feature": "AVG",
              "importance": 0.715131
            },
            {
              "feature": "ack_flag_number",
              "importance": 0.699537
            },
            {
              "feature": "Min",
              "importance": 0.533591
            },
            {
              "feature": "syn_flag_number",
              "importance": 0.523086
            },
            {
              "feature": "ICMP",
              "importance": 0.370109
            },
            {
              "feature": "flow_duration",
              "importance": 0.19883
            },
            {
              "feature": "syn_count",
              "importance": 0.185645
            }
          ],
          "rf": [
            {
              "feature": "Protocol Type",
              "importance": 0.209542
            },
            {
              "feature": "Min",
              "importance": 0.176736
            },
            {
              "feature": "Max",
              "importance": 0.155923
            },
            {
              "feature": "AVG",
              "importance": 0.148102
            },
            {
              "feature": "Tot size",
              "importance": 0.123254
            },
            {
              "feature": "Tot sum",
              "importance": 0.064878
            },
            {
              "feature": "TCP",
              "importance": 0.035197
            },
            {
              "feature": "Covariance",
              "importance": 0.017222
            },
            {
              "feature": "UDP",
              "importance": 0.015059
            },
            {
              "feature": "Header_Length",
              "importance": 0.014151
            }
          ],
          "xgb": [
            {
              "feature": "Protocol Type",
              "importance": 0.900505
            },
            {
              "feature": "Variance",
              "importance": 0.029802
            },
            {
              "feature": "Max",
              "importance": 0.020149
            },
            {
              "feature": "Number",
              "importance": 0.008304
            },
            {
              "feature": "Tot size",
              "importance": 0.005912
            },
            {
              "feature": "Min",
              "importance": 0.005072
            },
            {
              "feature": "Tot sum",
              "importance": 0.003928
            },
            {
              "feature": "HTTPS",
              "importance": 0.003867
            },
            {
              "feature": "Covariance",
              "importance": 0.002939
            },
            {
              "feature": "rst_count",
              "importance": 0.002617
            }
          ],
          "lgbm": [
            {
              "feature": "Protocol Type",
              "importance": 1298
            },
            {
              "feature": "Duration",
              "importance": 1091
            },
            {
              "feature": "Rate",
              "importance": 1074
            },
            {
              "feature": "flow_duration",
              "importance": 917
            },
            {
              "feature": "Header_Length",
              "importance": 882
            },
            {
              "feature": "Variance",
              "importance": 879
            },
            {
              "feature": "Max",
              "importance": 807
            },
            {
              "feature": "Tot size",
              "importance": 779
            },
            {
              "feature": "Covariance",
              "importance": 762
            },
            {
              "feature": "Min",
              "importance": 742
            }
          ]
        }
      }
    },
    "attack_metadata": {
      "run_timestamp": "2026-06-19T17:43:15.278299+00:00",
      "attacks_processed": [
        "ddos_icmp",
        "ddos_syn",
        "mirai_greeth"
      ],
      "attacks": {
        "ddos_icmp": {
          "attack": "ddos_icmp",
          "label_patterns": [
            "DDoS-ICMP_Flood",
            "ICMP_Flood"
          ],
          "features": [
            "Header_Length",
            "Duration",
            "syn_flag_number",
            "ack_flag_number",
            "syn_count",
            "urg_count",
            "rst_count",
            "HTTPS",
            "UDP",
            "ICMP",
            "Tot sum",
            "Min",
            "Max",
            "AVG",
            "Tot size",
            "Covariance",
            "Variance",
            "Protocol Type",
            "Rate",
            "flow_duration",
            "Number",
            "TCP"
          ],
          "base_features": [
            "Header_Length",
            "Duration",
            "syn_flag_number",
            "ack_flag_number",
            "syn_count",
            "urg_count",
            "rst_count",
            "HTTPS",
            "UDP",
            "ICMP",
            "Tot sum",
            "Min",
            "Max",
            "AVG",
            "Tot size",
            "Covariance",
            "Variance"
          ],
          "extra_features": [
            "Protocol Type",
            "Rate",
            "flow_duration",
            "Number",
            "TCP"
          ],
          "log1p_applied": [
            "Rate",
            "Number",
            "Covariance",
            "flow_duration"
          ],
          "zero_preserved": [
            "flow_duration"
          ],
          "rows": {
            "train_total": 5491971,
            "train_attack": 848088,
            "train_benign": 4643883,
            "validation": 1176851,
            "test": 1176851
          },
          "numeric_fill_stats": {
            "Header_Length": {
              "max": 9890104.9,
              "min": 0.0,
              "median": 54.0
            },
            "Duration": {
              "max": 255.0,
              "min": 0.0,
              "median": 64.0
            },
            "syn_flag_number": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "ack_flag_number": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "syn_count": {
              "max": 12.61,
              "min": 0.0,
              "median": 0.0
            },
            "urg_count": {
              "max": 4167.7,
              "min": 0.0,
              "median": 0.0
            },
            "rst_count": {
              "max": 9586.5,
              "min": 0.0,
              "median": 0.0
            },
            "HTTPS": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "UDP": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "ICMP": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "Tot sum": {
              "max": 108933.8,
              "min": 42.0,
              "median": 567.0
            },
            "Min": {
              "max": 4702.7,
              "min": 42.0,
              "median": 54.0
            },
            "Max": {
              "max": 30474.0,
              "min": 42.0,
              "median": 54.0
            },
            "AVG": {
              "max": 11600.474325396824,
              "min": 42.0,
              "median": 54.0
            },
            "Tot size": {
              "max": 13098.0,
              "min": 42.0,
              "median": 54.0
            },
            "Covariance": {
              "max": 96708369.53422824,
              "min": 0.0,
              "median": 0.0
            },
            "Variance": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "Protocol Type": {
              "max": 47.0,
              "min": 0.0,
              "median": 6.0
            },
            "Rate": {
              "max": 8388608.0,
              "min": 0.0,
              "median": 15.785982581728124
            },
            "flow_duration": {
              "max": 99435.76178212166,
              "min": 0.0,
              "median": 0.0
            },
            "Number": {
              "max": 15.0,
              "min": 1.0,
              "median": 9.5
            },
            "TCP": {
              "max": 1.0,
              "min": 0.0,
              "median": 1.0
            }
          },
          "scaler_path": "C:\\Users\\Suhas Raghavendra\\Desktop\\Main EL\\processed_ciciot23\\attack_specific\\ddos_icmp\\scaler.pkl"
        },
        "ddos_syn": {
          "attack": "ddos_syn",
          "label_patterns": [
            "DDoS-SYN_Flood",
            "SYN_Flood"
          ],
          "features": [
            "Header_Length",
            "Duration",
            "syn_flag_number",
            "ack_flag_number",
            "syn_count",
            "urg_count",
            "rst_count",
            "HTTPS",
            "UDP",
            "ICMP",
            "Tot sum",
            "Min",
            "Max",
            "AVG",
            "Tot size",
            "Covariance",
            "Variance",
            "TCP",
            "Rate",
            "flow_duration",
            "fin_flag_number",
            "psh_flag_number",
            "Number"
          ],
          "base_features": [
            "Header_Length",
            "Duration",
            "syn_flag_number",
            "ack_flag_number",
            "syn_count",
            "urg_count",
            "rst_count",
            "HTTPS",
            "UDP",
            "ICMP",
            "Tot sum",
            "Min",
            "Max",
            "AVG",
            "Tot size",
            "Covariance",
            "Variance"
          ],
          "extra_features": [
            "TCP",
            "Rate",
            "flow_duration",
            "fin_flag_number",
            "psh_flag_number",
            "Number"
          ],
          "log1p_applied": [
            "Rate",
            "Number",
            "Covariance",
            "flow_duration"
          ],
          "zero_preserved": [
            "flow_duration"
          ],
          "rows": {
            "train_total": 5491971,
            "train_attack": 716226,
            "train_benign": 4775745,
            "validation": 1176851,
            "test": 1176851
          },
          "numeric_fill_stats": {
            "Header_Length": {
              "max": 9890104.9,
              "min": 0.0,
              "median": 54.0
            },
            "Duration": {
              "max": 255.0,
              "min": 0.0,
              "median": 64.0
            },
            "syn_flag_number": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "ack_flag_number": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "syn_count": {
              "max": 12.61,
              "min": 0.0,
              "median": 0.0
            },
            "urg_count": {
              "max": 4167.7,
              "min": 0.0,
              "median": 0.0
            },
            "rst_count": {
              "max": 9586.5,
              "min": 0.0,
              "median": 0.0
            },
            "HTTPS": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "UDP": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "ICMP": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "Tot sum": {
              "max": 108933.8,
              "min": 42.0,
              "median": 567.0
            },
            "Min": {
              "max": 4702.7,
              "min": 42.0,
              "median": 54.0
            },
            "Max": {
              "max": 30474.0,
              "min": 42.0,
              "median": 54.0
            },
            "AVG": {
              "max": 11600.474325396824,
              "min": 42.0,
              "median": 54.0
            },
            "Tot size": {
              "max": 13098.0,
              "min": 42.0,
              "median": 54.0
            },
            "Covariance": {
              "max": 96708369.53422824,
              "min": 0.0,
              "median": 0.0
            },
            "Variance": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "TCP": {
              "max": 1.0,
              "min": 0.0,
              "median": 1.0
            },
            "Rate": {
              "max": 8388608.0,
              "min": 0.0,
              "median": 15.785982581728124
            },
            "flow_duration": {
              "max": 99435.76178212166,
              "min": 0.0,
              "median": 0.0
            },
            "fin_flag_number": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "psh_flag_number": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "Number": {
              "max": 15.0,
              "min": 1.0,
              "median": 9.5
            }
          },
          "scaler_path": "C:\\Users\\Suhas Raghavendra\\Desktop\\Main EL\\processed_ciciot23\\attack_specific\\ddos_syn\\scaler.pkl"
        },
        "mirai_greeth": {
          "attack": "mirai_greeth",
          "label_patterns": [
            "Mirai-Greeth_flood"
          ],
          "features": [
            "Header_Length",
            "Duration",
            "syn_flag_number",
            "ack_flag_number",
            "syn_count",
            "urg_count",
            "rst_count",
            "HTTPS",
            "UDP",
            "ICMP",
            "Tot sum",
            "Min",
            "Max",
            "AVG",
            "Tot size",
            "Covariance",
            "Variance",
            "Protocol Type",
            "Rate",
            "flow_duration",
            "Number",
            "TCP"
          ],
          "base_features": [
            "Header_Length",
            "Duration",
            "syn_flag_number",
            "ack_flag_number",
            "syn_count",
            "urg_count",
            "rst_count",
            "HTTPS",
            "UDP",
            "ICMP",
            "Tot sum",
            "Min",
            "Max",
            "AVG",
            "Tot size",
            "Covariance",
            "Variance"
          ],
          "extra_features": [
            "Protocol Type",
            "Rate",
            "flow_duration",
            "Number",
            "TCP"
          ],
          "log1p_applied": [
            "Rate",
            "Number",
            "Covariance",
            "flow_duration"
          ],
          "zero_preserved": [
            "flow_duration"
          ],
          "rows": {
            "train_total": 5491971,
            "train_attack": 116133,
            "train_benign": 5375838,
            "validation": 1176851,
            "test": 1176851
          },
          "numeric_fill_stats": {
            "Header_Length": {
              "max": 9890104.9,
              "min": 0.0,
              "median": 54.0
            },
            "Duration": {
              "max": 255.0,
              "min": 0.0,
              "median": 64.0
            },
            "syn_flag_number": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "ack_flag_number": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "syn_count": {
              "max": 12.61,
              "min": 0.0,
              "median": 0.0
            },
            "urg_count": {
              "max": 4167.7,
              "min": 0.0,
              "median": 0.0
            },
            "rst_count": {
              "max": 9586.5,
              "min": 0.0,
              "median": 0.0
            },
            "HTTPS": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "UDP": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "ICMP": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "Tot sum": {
              "max": 108933.8,
              "min": 42.0,
              "median": 567.0
            },
            "Min": {
              "max": 4702.7,
              "min": 42.0,
              "median": 54.0
            },
            "Max": {
              "max": 30474.0,
              "min": 42.0,
              "median": 54.0
            },
            "AVG": {
              "max": 11600.474325396824,
              "min": 42.0,
              "median": 54.0
            },
            "Tot size": {
              "max": 13098.0,
              "min": 42.0,
              "median": 54.0
            },
            "Covariance": {
              "max": 96708369.53422824,
              "min": 0.0,
              "median": 0.0
            },
            "Variance": {
              "max": 1.0,
              "min": 0.0,
              "median": 0.0
            },
            "Protocol Type": {
              "max": 47.0,
              "min": 0.0,
              "median": 6.0
            },
            "Rate": {
              "max": 8388608.0,
              "min": 0.0,
              "median": 15.785982581728124
            },
            "flow_duration": {
              "max": 99435.76178212166,
              "min": 0.0,
              "median": 0.0
            },
            "Number": {
              "max": 15.0,
              "min": 1.0,
              "median": 9.5
            },
            "TCP": {
              "max": 1.0,
              "min": 0.0,
              "median": 1.0
            }
          },
          "scaler_path": "C:\\Users\\Suhas Raghavendra\\Desktop\\Main EL\\processed_ciciot23\\attack_specific\\mirai_greeth\\scaler.pkl"
        }
      },
      "log1p_policy": {
        "features": [
          "Rate",
          "Number",
          "Covariance"
        ],
        "zero_preserve_features": [
          "flow_duration"
        ],
        "note": "flow_duration zeros are kept as-is (they are the DDoS attack signal). Only positive flow_duration values get log1p applied."
      }
    },
    "signal_strengths": {
      "ddos_icmp": [
        {
          "feature": "ICMP",
          "signal": 97,
          "importance": "critical"
        },
        {
          "feature": "Protocol Type",
          "signal": 95,
          "importance": "critical"
        },
        {
          "feature": "Rate",
          "signal": 93,
          "importance": "critical"
        },
        {
          "feature": "Tot sum",
          "signal": 91,
          "importance": "critical"
        },
        {
          "feature": "Srate",
          "signal": 88,
          "importance": "high"
        },
        {
          "feature": "Number",
          "signal": 85,
          "importance": "high"
        },
        {
          "feature": "flow_duration",
          "signal": 82,
          "importance": "high"
        },
        {
          "feature": "AVG",
          "signal": 78,
          "importance": "high"
        },
        {
          "feature": "Header_Length",
          "signal": 74,
          "importance": "high"
        },
        {
          "feature": "Std/Variance",
          "signal": 62,
          "importance": "medium"
        },
        {
          "feature": "Magnitue",
          "signal": 60,
          "importance": "medium"
        },
        {
          "feature": "TCP/UDP/HTTPS",
          "signal": 58,
          "importance": "medium"
        }
      ],
      "ddos_syn": [
        {
          "feature": "syn_flag_number",
          "signal": 96,
          "importance": "critical"
        },
        {
          "feature": "syn_count",
          "signal": 94,
          "importance": "critical"
        },
        {
          "feature": "ack_flag_number",
          "signal": 93,
          "importance": "critical"
        },
        {
          "feature": "TCP",
          "signal": 90,
          "importance": "critical"
        },
        {
          "feature": "Rate/Srate",
          "signal": 89,
          "importance": "critical"
        },
        {
          "feature": "ack_count",
          "signal": 80,
          "importance": "high"
        },
        {
          "feature": "fin_flag_number",
          "signal": 77,
          "importance": "high"
        },
        {
          "feature": "flow_duration",
          "signal": 76,
          "importance": "high"
        },
        {
          "feature": "Header_Length",
          "signal": 73,
          "importance": "high"
        },
        {
          "feature": "psh_flag_number",
          "signal": 64,
          "importance": "medium"
        },
        {
          "feature": "rst_flag_number",
          "signal": 58,
          "importance": "medium"
        },
        {
          "feature": "Number/Tot sum",
          "signal": 70,
          "importance": "medium"
        }
      ],
      "mirai_greeth": [
        {
          "feature": "Protocol Type",
          "signal": 97,
          "importance": "critical"
        },
        {
          "feature": "Header_Length",
          "signal": 94,
          "importance": "critical"
        },
        {
          "feature": "Tot sum/Magnitue",
          "signal": 92,
          "importance": "critical"
        },
        {
          "feature": "AVG",
          "signal": 91,
          "importance": "critical"
        },
        {
          "feature": "Rate/Srate",
          "signal": 85,
          "importance": "high"
        },
        {
          "feature": "Max/Min",
          "signal": 82,
          "importance": "high"
        },
        {
          "feature": "Covariance",
          "signal": 78,
          "importance": "high"
        },
        {
          "feature": "Duration",
          "signal": 75,
          "importance": "high"
        },
        {
          "feature": "Variance/Radius",
          "signal": 65,
          "importance": "medium"
        },
        {
          "feature": "Number",
          "signal": 62,
          "importance": "medium"
        },
        {
          "feature": "TCP/UDP/ICMP",
          "signal": 80,
          "importance": "medium"
        }
      ]
    }
  },
  "graph": {
    "graph_summary": {
      "nodes": 20,
      "edges": 190,
      "density": 1.0,
      "avg_degree": 19.0,
      "max_degree": 19,
      "is_connected": true,
      "num_connected_components": 1
    },
    "partition_summary": {
      "n_partitions": 4,
      "cross_partition_edges": 54,
      "per_partition": {
        "0": {
          "node_count": 1,
          "nodes": [
            12
          ],
          "internal_edges": 0,
          "total_flows": 6801,
          "attack_flows": 2325,
          "attack_ratio": 0.3418614909572122
        },
        "1": {
          "node_count": 17,
          "nodes": [
            0,
            1,
            2,
            3,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            14,
            15,
            16,
            17,
            18,
            19
          ],
          "internal_edges": 136,
          "total_flows": 1162999,
          "attack_flows": 1142404,
          "attack_ratio": 0.9822914723056512
        },
        "2": {
          "node_count": 1,
          "nodes": [
            13
          ],
          "internal_edges": 0,
          "total_flows": 2507,
          "attack_flows": 2505,
          "attack_ratio": 0.9992022337455125
        },
        "3": {
          "node_count": 1,
          "nodes": [
            5
          ],
          "internal_edges": 0,
          "total_flows": 4544,
          "attack_flows": 2098,
          "attack_ratio": 0.46170774647887325
        }
      }
    },
    "spectral_summary": {
      "split": "validation",
      "top_k": 8,
      "laplacian_type": "normalized",
      "new_columns": [
        "spectral_eigen_0",
        "spectral_eigen_1",
        "spectral_eigen_2",
        "spectral_eigen_3",
        "spectral_eigen_4",
        "spectral_eigen_5",
        "spectral_eigen_6",
        "spectral_eigen_7",
        "spectral_proj_0",
        "spectral_proj_1",
        "spectral_proj_2",
        "spectral_proj_3",
        "spectral_proj_4",
        "spectral_proj_5",
        "spectral_proj_6",
        "spectral_proj_7",
        "fiedler_value",
        "partition_id"
      ],
      "partitions_processed": 4,
      "per_partition_summary": {
        "0": {
          "node_count": 1,
          "eigenvalues": [
            1.0
          ],
          "fiedler_value": 0.0
        },
        "1": {
          "node_count": 17,
          "eigenvalues": [
            4.501383225390492e-16,
            1.0595664328963974,
            1.0625520524508612,
            1.0626052475129566,
            1.062627495345095,
            1.06262847379802,
            1.0626539719435022,
            1.0626924274951035
          ],
          "fiedler_value": 1.0595664328963974
        },
        "2": {
          "node_count": 1,
          "eigenvalues": [
            1.0
          ],
          "fiedler_value": 0.0
        },
        "3": {
          "node_count": 1,
          "eigenvalues": [
            1.0
          ],
          "fiedler_value": 0.0
        }
      }
    },
    "eigenvalues": [
      0.0,
      1.059566,
      1.062552,
      1.062605,
      1.062627,
      1.062628,
      1.062654,
      1.062692
    ],
    "fiedler_value": 1.0595664328963974,
    "top_k": 8,
    "laplacian_type": "normalized",
    "new_spectral_cols": [
      "spectral_eigen_0",
      "spectral_eigen_1",
      "spectral_eigen_2",
      "spectral_eigen_3",
      "spectral_eigen_4",
      "spectral_eigen_5",
      "spectral_eigen_6",
      "spectral_eigen_7",
      "spectral_proj_0",
      "spectral_proj_1",
      "spectral_proj_2",
      "spectral_proj_3",
      "spectral_proj_4",
      "spectral_proj_5",
      "spectral_proj_6",
      "spectral_proj_7",
      "fiedler_value",
      "partition_id"
    ]
  },
  "federated": {
    "rounds": [
      {
        "round": 1,
        "global_accuracy": 0.9882,
        "global_f1": 0.9939,
        "n_clients": 2,
        "total_samples": 3334411
      },
      {
        "round": 2,
        "global_accuracy": 0.9979,
        "global_f1": 0.9989,
        "n_clients": 2,
        "total_samples": 3334411
      }
    ],
    "final_eval": {
      "model_path": "C:\\Users\\Suhas Raghavendra\\Desktop\\Main EL\\federated_artifacts\\global_model_final.pkl",
      "n_trees": 100,
      "n_features": 17,
      "test_samples": 1176851,
      "accuracy": 0.9930832365354663,
      "f1": 0.9964583442106546,
      "macro_f1": 0.9246864739737093,
      "roc_auc": 0.9982986003253153,
      "pr_auc": 0.9999569770650629,
      "confusion_matrix": [
        [
          23601,
          4108
        ],
        [
          4032,
          1145110
        ]
      ]
    },
    "n_rounds": 2,
    "n_clients": 2,
    "privacy_note": "Additive masking via Paillier-style secure aggregation implemented.",
    "fl_architecture": {
      "framework": "Flower (flwr)",
      "strategy": "FedAvg (custom RF parameter aggregation)",
      "clients": 2,
      "rounds": 2
    }
  },
  "matrix": {
    "baseline_acc": 0.9879,
    "baseline_f1": 0.9938,
    "matrix_acc": 0.9947,
    "matrix_f1": 0.9973,
    "combined_acc": 0.9946,
    "combined_f1": 0.9972,
    "matrix_features": 17,
    "model_comparison": [
      {
        "name": "Baseline RF",
        "features": 17,
        "accuracy": 0.9879,
        "f1": 0.9938
      },
      {
        "name": "Spectral RF",
        "features": 35,
        "accuracy": 0.9933,
        "f1": 0.9966
      },
      {
        "name": "Matrix RF",
        "features": 34,
        "accuracy": 0.9947,
        "f1": 0.9973
      },
      {
        "name": "Combined RF (base+spectral+matrix)",
        "features": 51,
        "accuracy": 0.9946,
        "f1": 0.9972
      }
    ],
    "feature_importance": [
      {
        "feature": "Min",
        "importance": 0.228,
        "group": "base"
      },
      {
        "feature": "AVG",
        "importance": 0.179,
        "group": "base"
      },
      {
        "feature": "Tot size",
        "importance": 0.157,
        "group": "base"
      },
      {
        "feature": "spectral_eigen_0",
        "importance": 0.095,
        "group": "spectral"
      },
      {
        "feature": "spectral_proj_0",
        "importance": 0.088,
        "group": "spectral"
      },
      {
        "feature": "Protocol Type",
        "importance": 0.071,
        "group": "base"
      },
      {
        "feature": "Tot sum",
        "importance": 0.065,
        "group": "base"
      },
      {
        "feature": "matrix_feat_0",
        "importance": 0.057,
        "group": "matrix"
      },
      {
        "feature": "matrix_feat_1",
        "importance": 0.042,
        "group": "matrix"
      },
      {
        "feature": "ICMP",
        "importance": 0.018,
        "group": "base"
      }
    ]
  }
};
