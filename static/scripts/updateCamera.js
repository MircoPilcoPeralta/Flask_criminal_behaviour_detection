function updateCameraClousure(){
    const detectionModelRef = document.getElementById("detection-model");
    const inferenceValueRef = document.getElementById("inference");
    const camerasList = loadItem("connected_cameras");
    let currentCameraData = null;
    let currentId = -1;

    function loadCameraData (cameraId) {
        currentId = Number(cameraId);
        currentCameraData = camerasList.find((camera)=> camera.id == currentId);

        detectionModelRef.value = currentCameraData.activeModel;
        inferenceValueRef.value = currentCameraData.inferencePercentage;
    }



    function updateCamera(){
        if(!currentCameraData){
            return;
        }

        currentCameraData["activeModel"] = detectionModelRef.value;
        currentCameraData["inferencePercentage"] = inferenceValueRef.value;
        // camera.relevantItems = 
    
        updateCameraInLocalStorage();
    
    
        fetch(`/surveillance/camera/${currentId}/config`, {
            method: 'PATCH',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentCameraData)
          })

        setTimeout(()=>{
            location.reload();
        }, 1000)
    }

    function updateCameraInLocalStorage() {
        const cameraIndex = camerasList.findIndex((camera)=> camera.id == currentId );

        if(cameraIndex >= 0){
            const updatedCamera = camerasList[cameraIndex];
            updatedCamera.activeModel = detectionModelRef.value;
            updatedCamera.inferencePercentage = inferenceValueRef.value;
        }
        
        saveConnectedCamerasList(camerasList);
    }



    return { loadCameraData, updateCameraInLocalStorage, updateCamera  }
}























