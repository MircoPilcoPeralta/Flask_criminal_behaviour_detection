class Modal{
    constructor(modalReferenceId){
        this.modalElement = document.getElementById(modalReferenceId);
    }

    openModal() {
      try {
        this.modalElement.style.display = "block";
      } catch (error) {
        console.log("Error: ", error);
      }
    }

    closeModal() {
        try {
            this.modalElement.style.display = "none";
        } catch (error) {
          console.log("Error: ", error);
        }
      }


    setTargetModal (modalReferenceId) {
        this.modalElement = document.getElementById(modalReferenceId);
      };
}