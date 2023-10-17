const API_URL = "http://localhost:3000";
const NO_EMAIL_INSERTED = "Debe ingresar su correo electrónico";
const NO_PASSWORD_INSERTED = "Debe ingresar su contraseña";

const loginErrorContainer = document.getElementById("login-error-container");
const loginError = document.getElementById("login-error");

const registerErrorContainer = document.getElementById("register-error-container");
const registerError = document.getElementById("register-error");

const onLogin = (event) => {
  event.preventDefault();
  checkLoginForm();
};

const onRegister = (event) => {
  event.preventDefault();
  checkRegisterForm();
}

const showFormError = (elementContainer, textElement, message) =>{
  textElement.textContent = message;
  elementContainer.style.visibility = "visible";
  setTimeout(()=>{
    elementContainer.style.visibility = "hidden";
  },2000)
}


const checkLoginForm = async () => {
  const emailField = document.getElementById("email");
  const passwordField = document.getElementById("password");

  if (emailField.value.length === 0  || passwordField.value.length === 0 ) {
    showFormError(loginErrorContainer, loginError, "Debe ingresar su email y contraseña");
    return;
  } 

    const postSettings = {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email: emailField.value, password: passwordField.value  }),
    };

    const fetchResults = await fetch(API_URL+"/auth/login", postSettings);
    const response = await fetchResults.json();

    if (response.statusCode === 400) {
      showFormError(loginErrorContainer, loginError, response.message);
    }

    if(response.statusCode === 200){
      localStorage.setItem("token", response.data.access_token);
      setTimeout(()=>{window.location.href = "/";}, 1000)
      
    }

};

const checkRegisterForm = async () =>{
  const fullNameField = document.getElementById("name-register-input");
  const emailField = document.getElementById("email-register-input");
  const passwordField = document.getElementById("password-register-input");
  const confirmPasswordField = document.getElementById("confirm-password-register-input");
  
  if (    fullNameField.value.length           === 0  
      || emailField.value.length           === 0 
      || passwordField.value.length        === 0 
      ) {
    showFormError(registerErrorContainer, registerError, "Debe llenar todos los campos");
    return;
  }

  if ( passwordField.value !== confirmPasswordField.value) {
  showFormError(registerErrorContainer, registerError, "Debe repetir la misma contraseña");
    return;
  } 

  const postSettings = {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ 
      fullName: fullNameField.value, 
      email: emailField.value,
      password: passwordField.value,
      confirmPassword: confirmPasswordField.value
     }),
  };

  const fetchResults = await fetch(API_URL+"/auth/register", postSettings);
  const response = await fetchResults.json();

  if (response.statusCode === 400) {
    showFormError(registerErrorContainer, registerError, response.message);
  }

  if(response.statusCode === 200){
    localStorage.setItem("token", response.data.access_token);
    setTimeout(()=>{window.location.href = "/";}, 1000)
  }

}






