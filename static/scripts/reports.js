const token = localStorage.getItem("token") || "";

const getReports = async() =>{

    const loggedUser = decodeJWT(token);

    if(!loggedUser){
        return [];
    }

    const apiUrl = `http://localhost:3000/api/v1/report?userId=${loggedUser.id}`;

    const requestOptions ={
        method: "GET",
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
        }
    }

    const response = await fetch(apiUrl, requestOptions)
    const { data } = await response.json();
    return data;
}


const saveReport = (report) =>{
    const loggedUser = decodeJWT(token);
}

const renderReports = async() =>{
    const reportsContainer = document.querySelector(".reports__table-body");
    let updatedHtmlContent = ""
    reportsContainer.innerHTML = "";

    const reports = await getReports();
    console.log(reports)

    reports.forEach(report => {
        updatedHtmlContent += `<tr>
                        <td> ${report.cameraName} </td>
                        <td> ${report.inference} % </td>
                        <td> ${report.message } </td>
                        <td> <img src="${report.imageURL}" height="100"  alt="${report.message}" />  </td>
                        <td> ${report.date} ${report.time} </td>
                        <td> <button class="btn-red" onClick="deleteReport('${report.id}')"> Eliminar </button> </td>
                        </tr> `
    });


        reportsContainer.innerHTML = updatedHtmlContent;

}




const deleteReport = async(reportId) =>{
    const url = `http://localhost:3000/api/v1/report/${reportId}`;

    const requestOptions ={
        method: "DELETE",
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
        }
    }
    await fetch(url, requestOptions);
    await renderReports();
}



renderReports();