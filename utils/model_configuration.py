class ModelConfiguration:

    configuration = {
        "models": [
            {
                "name": "ssd-object_detection",
                "enable": False,
                "inference_percentage": 60,
                "relevant_items": [ "pistola", "cuchillo" ],
            },
            {
                "name": "multipose-criminal_behaviour_detection",
                "enable": False,
                "inference_percentage": 60,
                "relevant_actions": ["correr", "golpeando", "apuntando"]
            },
            {
                "name": "haar_cascade_face_detection",
                "enable": True,
                "inference_percentage": 60
            }
        ]
    }

    def __init__(self, camera_id):
        self.camera_id = camera_id;
        return None;
    

    def get_model_configuration(self, model_name): 
        for model in self.configuration["models"]:
            if model["name"] == model_name:
                return model;