const noLoggedUserPaths = ["/auth/login", "/auth/register"];
const loggedUserPaths = ["/", "/surveillance/no-cameras-added", "/surveillance/one-camera-image", "/reports"]


const closeSessionBtn = document.getElementById("close-sesión-btn");
closeSessionBtn.addEventListener("click",()=>{
  localStorage.removeItem("token");
  location.reload();
})

const isAuthenticated = async() => {
    const token = localStorage.getItem("token") || "";
    const apiUrl = "http://localhost:3000/api/v1/user/validate-token";

    const requestOptions ={
        method: "GET",
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
        }
    }
    
    const response = await fetch(apiUrl, requestOptions)
    const responseJson = await response.json();
    return (responseJson.statusCode === 200);
}

const verifySession = async() =>{
    if(await isAuthenticated() && noLoggedUserPaths.includes(location.pathname)){
        location.replace("/");
    }

    if(!await isAuthenticated() && loggedUserPaths.includes(location.pathname)){
        location.replace("/auth/login?=no-authenticated");
    }
}


verifySession();