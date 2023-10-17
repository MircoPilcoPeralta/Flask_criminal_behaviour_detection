const loadItem = ( itemName ) =>{
    item = localStorage.getItem(itemName)
    item? item : "[]";
}

const saveConnectedCamerasList = ( state ) =>{
    localStorage.setItem("connected_cameras", JSON.stringify(state));
}


const getToken = () => {
    return localStorage.getItem("token") | "";
}


