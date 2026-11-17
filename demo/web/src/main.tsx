import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/lato/300.css";
import "@fontsource/lato/400.css";
import "@fontsource/lato/700.css";
import "@fontsource/montserrat/300.css";
import "@fontsource/montserrat/400.css";
import "./index.css";
import "./panel/knob.css";
import { App } from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
