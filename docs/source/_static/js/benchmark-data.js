window.BENCHMARK_DATA = {
  "subtitle": "192 CPUs, 749 GB RAM",
  "groups": {
    "Basic workflow": {
      "hardware": "cpu",
      "bars": {
        "brisc": 175.0,
        "scanpy": 20342.42,
        "seurat": 41217.8
      }
    },
    "Label transfer": {
      "hardware": "cpu",
      "bars": {
        "brisc": 33.97,
        "scanpy": 3260.71,
        "seurat": 10097.32
      }
    },
    "Pseudobulk differential expression": {
      "hardware": "cpu",
      "bars": {
        "brisc": 27.12,
        "scanpy": 1092.36,
        "seurat": 4469.89
      }
    },
    "Basic workflow · CPU vs GPU": {
      "hardware": "gpu",
      "note": "96 CPUs, 4× H100 GPU, 752 GB RAM",
      "bars": {
        "brisc": 332.38,
        "rapids": 561.62
      }
    }
  }
};
