const destination = new URLSearchParams(location.search).get("next") === "/test-voice" ? "/test-voice" : "/app";
const form = document.getElementById("login-form");
const errorEl = document.getElementById("login-error");

if (sessionStorage.getItem("token")) {
  location.replace(destination);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorEl.hidden = true;
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;
  const res = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    errorEl.textContent = data.error || "ورود ناموفق بود";
    errorEl.hidden = false;
    return;
  }
  sessionStorage.setItem("token", data.token);
  sessionStorage.setItem("username", data.username);
  location.replace(destination);
});
