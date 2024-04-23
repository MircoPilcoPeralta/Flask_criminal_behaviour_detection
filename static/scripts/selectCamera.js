const errorMessageContainer = document.querySelector(".form-error-message");
const messageParagraph = document.getElementById("camera-form-error");

const loadItem = ( itemName ) =>{
    item = localStorage.getItem(itemName)
    if(!item){
        return []; 
    }
    return JSON.parse(item);
}

const saveConnectedCamerasList = ( state ) =>{
    localStorage.setItem("connected_cameras", JSON.stringify(state));
}

const showFormError = (elementContainer, textElement, message) =>{
    textElement.textContent = message;
    elementContainer.classList.toggle("form-error-message-active");

    setTimeout(()=>{
        elementContainer.classList.toggle("form-error-message-active");
    }, 2500)
}


const registerCamera = () =>{
    const connectedCamerasList = loadItem("connected_cameras") ;
    const id = document.getElementById("camera-id");
    const cameraName = document.getElementById("camera-name");
    const cameraModel = document.getElementById("camera-model");

    if(cameraName.value.length === 0){
        showFormError(errorMessageContainer,messageParagraph,"El nombre de la cámara no puede estar vacio o contener caracteres especiales")
        return;
    }

    if(isNaN(+(id.value)) ){
        showFormError(errorMessageContainer,messageParagraph,"Debe seleccionar un modelo de cámara")
        return;  
    }

    const isCameraImageSelected = connectedCamerasList.find(
        (cameraImageSelected) => cameraImageSelected.id === id.value 
    );
 
    if(isCameraImageSelected){
        showFormError(errorMessageContainer,messageParagraph,"La cámara seleccionada ya se encuentra en uso")
        return;
    }

    connectedCamerasList.push({ 
        id: id.value ,
        cameraName: cameraName.value, 
        model: id.value,
        activated: true,
        activeModel: "No_model" ,
        relevantItems: [],
        inferencePercentage : 0
    })

    saveConnectedCamerasList(connectedCamerasList);
    sendUpdatedConnectedCamerasList();
    setTimeout(()=>{
        location.reload();
    }, 1000)
    
}

const sendUpdatedConnectedCamerasList = ( ) =>{
    const conected_cameras_object = {
        cameras: loadItem("connected_cameras"),
      };

    fetch('/surveillance/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(conected_cameras_object)
      })
}


const loadCurrentConnectedCameras = (page) =>{
    const conected_cameras_list = loadItem("connected_cameras");

    const conected_cameras_object = {
        cameras: conected_cameras_list,
      };

    fetch('/load_connected_cameras', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(conected_cameras_object)
      })
}