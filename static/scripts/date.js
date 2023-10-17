const getCurrentDate = () => {
    const currentDate = new Date();
    const day = currentDate.getDate().toString().padStart(2, '0');
    const month = (currentDate.getMonth() + 1).toString().padStart(2, '0'); 
    const year = currentDate.getFullYear().toString();
    return `Dia: ${day}/${month}/${year}`;
  }


  const getCurrentTime = () => {
    const currentDate = new Date();
    const hour = currentDate.getHours().toString().padStart(2, '0');
    const minute = currentDate.getMinutes().toString().padStart(2, '0');
    const second = currentDate.getSeconds().toString().padStart(2, '0');
    return `Hora: ${hour}:${minute}:${second}`;
  }


const infoSection = document.getElementById("info");

const showDateTime = () =>{
    setInterval(() => {
        infoSection.innerHTML =  ` <p> ${getCurrentDate()} </p> <p> ${getCurrentTime()} </p>`
    }, 1000);
}
try {
    showDateTime();
} catch (error) {
    console.log("Error al colocar la informacion de la fecha y hora en la sección de información,", error)
}



