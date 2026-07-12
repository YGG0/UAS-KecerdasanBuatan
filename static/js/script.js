// ======================================================
// Brain Tumor AI
// Rezha Achmad Muharam
// Faujan Alamsyah
// ======================================================

document.addEventListener("DOMContentLoaded", function () {

    // ===============================
    // PREVIEW GAMBAR
    // ===============================

    const input = document.querySelector("input[type='file']");

    if (input) {

        const preview = document.createElement("img");

        preview.id = "preview-image";

        preview.style.width = "350px";
        preview.style.margin = "25px auto";
        preview.style.display = "none";
        preview.style.borderRadius = "15px";
        preview.style.boxShadow = "0 10px 25px rgba(0,0,0,.15)";

        input.parentNode.insertBefore(preview, input.nextSibling);

        input.addEventListener("change", function () {

            if (!this.files.length) return;

            const file = this.files[0];

            preview.src = URL.createObjectURL(file);

            preview.style.display = "block";

        });

    }

    // ===============================
    // LOADING BUTTON
    // ===============================

    const form = document.querySelector("form");

    if (form) {

        form.addEventListener("submit", function () {

            const btn = form.querySelector("button");

            btn.innerHTML = "⏳ Sedang Menganalisis...";

            btn.disabled = true;

        });

    }

    // ===============================
    // ANIMASI PROGRESS BAR
    // ===============================

    const bars = document.querySelectorAll(".progress-bar");

    bars.forEach(bar => {

        const value = bar.style.width;

        bar.style.width = "0%";

        setTimeout(() => {

            bar.style.width = value;

        },300);

    });

    // ===============================
    // SMOOTH SCROLL
    // ===============================

    document.querySelectorAll('a[href^="#"]').forEach(anchor=>{

        anchor.addEventListener("click",function(e){

            e.preventDefault();

            document.querySelector(this.getAttribute("href"))
            .scrollIntoView({

                behavior:"smooth"

            });

        });

    });

    // ===============================
    // CARD ANIMATION
    // ===============================

    const cards = document.querySelectorAll(".card");

    const observer = new IntersectionObserver(entries=>{

        entries.forEach(entry=>{

            if(entry.isIntersecting){

                entry.target.style.opacity="1";

                entry.target.style.transform="translateY(0px)";

            }

        });

    });

    cards.forEach(card=>{

        card.style.opacity="0";

        card.style.transform="translateY(40px)";

        card.style.transition=".8s";

        observer.observe(card);

    });

    // ===============================
    // HERO COUNTER
    // ===============================

    const numbers=document.querySelectorAll(".stat-card h2");

    numbers.forEach(item=>{

        const target=item.innerText;

        if(target.includes("%")) return;

    });

});

// =====================================
// DRAG & DROP
// =====================================

const dropArea = document.getElementById("drop-area");
const input = document.getElementById("fileInput");
const preview = document.getElementById("preview");

if (dropArea && input && preview) {

    dropArea.onclick = () => input.click();

    input.onchange = () => {
        if (input.files.length > 0) {
            showImage(input.files[0]);
        }
    };

    dropArea.addEventListener("dragover", (e) => {

        e.preventDefault();

        dropArea.classList.add("dragover");

    });

    dropArea.addEventListener("dragleave", () => {

        dropArea.classList.remove("dragover");

    });

    dropArea.addEventListener("drop", (e) => {

        e.preventDefault();

        dropArea.classList.remove("dragover");

        input.files = e.dataTransfer.files;

        if (input.files.length > 0) {

            showImage(input.files[0]);

        }

    });

    function showImage(file){

        const reader = new FileReader();

        reader.onload = function(e){

            preview.src = e.target.result;

            preview.style.display = "block";

        }

        reader.readAsDataURL(file);

    }

}