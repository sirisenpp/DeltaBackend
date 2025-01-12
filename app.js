
const EMAILJS_CONFIG = {
    userID: "oAF0i4QiMqH1BOox5",
    serviceID: "service_fcatscb",
    templateID: "template_9rcj1jh",
};

emailjs.init(EMAILJS_CONFIG.userID);

document.addEventListener("DOMContentLoaded", () => {
    
    const contactBtn = document.querySelector(".contact-page__button");

    if (!contactBtn) {
        console.error("Contact button not found!");
        return;
    }

    const popupOverlay = document.createElement("div");
    popupOverlay.className = "contact-popup-overlay";
    popupOverlay.innerHTML = `
        <div class="contact-popup">
            <button class="close-btn">&times;</button>
            <h2>Contact Us</h2>
            <form id="contact-form">
                <label for="name">Name</label>
                <input type="text" id="name" name="name" placeholder="Your Name" required>
                <label for="email">Email</label>
                <input type="email" id="email" name="email" placeholder="Your Email" required>
                <label for="phone">Phone</label>
                <input type="tel" id="phone" name="phone" placeholder="Your Phone Number">
                <button type="submit">Submit</button>
            </form>
        </div>
    `;

    document.body.appendChild(popupOverlay);

    const closeBtn = popupOverlay.querySelector(".close-btn");

    contactBtn.addEventListener("click", () => popupOverlay.classList.add("active"));
    closeBtn.addEventListener("click", () => popupOverlay.classList.remove("active"));
    popupOverlay.addEventListener("click", (e) => {
        if (e.target === popupOverlay) {
            popupOverlay.classList.remove("active");
        }
    });

    const form = document.querySelector("#contact-form");
    form.addEventListener("submit", (event) => {
        event.preventDefault();

      
        const formData = new FormData(form);
        const data = {
            from_email: formData.get("email"),
            name: formData.get("name"),
            phone: formData.get("phone"),
        };

        console.log("Sending data to EmailJS:", data);

        emailjs.send(EMAILJS_CONFIG.serviceID, EMAILJS_CONFIG.templateID, data)
    .then((response) => {
        alert("Email sent successfully: " + JSON.stringify(response));
        popupOverlay.classList.remove("active");
        form.reset();
    })
    .catch((error) => {
        alert("Error sending email: " + JSON.stringify(error));
        alert("Error details: " + error.text); 
    });

    });
});
