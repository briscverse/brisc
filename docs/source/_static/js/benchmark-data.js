window.BENCHMARK_DATA = {
  "subtitle": "192 CPUs, 755 GB RAM",
  "groups": {
    "Basic workflow": {
      "hardware": "cpu",
      "bars": {
        "brisc": 179.4,
        "scanpy": 18791.91,
        "seurat": 41129.28
      }
    },
    "Label transfer": {
      "hardware": "cpu",
      "bars": {
        "brisc": 44.9,
        "scanpy": 2462.79,
        "seurat": 36234.71
      }
    },
    "Pseudobulk differential expression": {
      "hardware": "cpu",
      "bars": {
        "brisc": 26.64,
        "scanpy": 1075.53,
        "seurat": 4357.93
      }
    },
    "Basic workflow · CPU vs GPU": {
      "hardware": "gpu",
      "note": "96 CPUs, 4× H100 GPU, 752 GB RAM",
      "bars": {
        "brisc": 555.28,
        "rapids": 862.59
      }
    }
  }
};
