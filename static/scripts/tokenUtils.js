const decodeJWT = (token) => {
    try {
        const [, payloadBase64] = token.split('.');
        return JSON.parse(atob(payloadBase64));
    } catch (error) {
        return null;
    }
}


const getUserDataFromSavedToken = () =>{
    return decodeJWT(localStorage.getItem("token") || "");
}


