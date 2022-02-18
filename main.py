from cProfile import run
import runpy

runpy.run_module("src.pipelines.pipeline_001", run_name="__main__")
runpy.run_module("src.pipelines.pipeline_002", run_name="__main__")
runpy.run_module("src.models.xgboost_pipeline_002")