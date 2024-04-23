const removeCamera = (id) => {

  console.log(loadItem("connected_cameras"), id)

  const updated_connected_cameras = loadItem("connected_cameras").filter(
    (camera) => camera.id != id
  );

    saveConnectedCamerasList(updated_connected_cameras);
    
    fetch(`/surveillance/camera?id=${id}`, {
        method: 'delete'
      }).then(
        ()=> window.location.replace("/surveillance")
       );


};