PROJECT_NAME = "SIIM"

if PROJECT_NAME == "HPA":
    from project.hpa_project.hpa_config import *

elif PROJECT_NAME == "QUBO":
    from project.qubo_project.qubo_config import *

elif PROJECT_NAME == "HisCancer":
    from project.HisCancer_project.HisCancer_config import *

elif PROJECT_NAME == "IMet":
    from project.imet_project.imet_config import *

elif PROJECT_NAME == "SIIM":
    from project.siim_project.siim_config import *