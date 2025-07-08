import { useState } from "react";
import axios from "axios";
import {
  Container, TextField, Button, MenuItem, Typography, Paper, Box, Alert, CircularProgress
} from "@mui/material";
import { MdSend } from "react-icons/md";
import { postDataApi } from "./api";
import { toast } from "react-toastify";
import './App.css'

const openAlertBox = (status, msg) => {
  if (status === "success") {
    toast.success(msg);
  } else {
    toast.error(msg);
  }
};

function App() {
  const [form, setForm] = useState({
    Sex: "M",
    Length: "",
    Diameter: "",
    Height: "",
    Whole_weight: "",
    Shucked_weight: "",
    Viscera_weight: "",
    Shell_weight: ""
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const allFieldsFilled = Object.values(form).every((el) => el.trim() !== "");
  const isFormValid = allFieldsFilled;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResult(null);
    setError("");

    const payload = {
      ...form,
      Length: parseFloat(form.Length),
      Diameter: parseFloat(form.Diameter),
      Height: parseFloat(form.Height),
      Whole_weight: parseFloat(form.Whole_weight),
      Shucked_weight: parseFloat(form.Shucked_weight),
      Viscera_weight: parseFloat(form.Viscera_weight),
      Shell_weight: parseFloat(form.Shell_weight)
    };

    const res = await postDataApi("/", payload);

    if (res?.success === true) {
      openAlertBox("success", res?.message || "Dự đoán thành công!");
      setForm({
        Sex: "M",
        Length: "",
        Diameter: "",
        Height: "",
        Whole_weight: "",
        Shucked_weight: "",
        Viscera_weight: "",
        Shell_weight: ""
      });
      setResult(res.data);
    } else {
      openAlertBox("error", res?.data?.detail || "Lỗi không xác định");
      setError(res?.data?.detail || "Lỗi không xác định");
    }

    setIsLoading(false);
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 6 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h5" align="center" gutterBottom>
          Dự đoán tuổi Abalone
        </Typography>

        <form onSubmit={handleSubmit}>
          <TextField
            select
            label="Giới tính"
            name="Sex"
            value={form.Sex}
            onChange={handleChange}
            fullWidth
            margin="normal"
            disabled={isLoading}
          >
            <MenuItem value="M">M (Male)</MenuItem>
            <MenuItem value="F">F (Female)</MenuItem>
            <MenuItem value="I">I (Infant)</MenuItem>
          </TextField>

          {[
            { field: "Length", label: "Length (0.1 - 0.8)", min: 0.1, max: 0.8 },
            { field: "Diameter", label: "Diameter (0.1 - 0.6)", min: 0.1, max: 0.6 },
            { field: "Height", label: "Height (0.02 - 0.3)", min: 0.02, max: 0.3 },
            { field: "Whole_weight", label: "Whole weight (0.05 - 2.5)", min: 0.05, max: 2.5 },
            { field: "Shucked_weight", label: "Shucked weight (0.02 - 1.3)", min: 0.02, max: 1.3 },
            { field: "Viscera_weight", label: "Viscera weight (0.01 - 0.5)", min: 0.01, max: 0.5 },
            { field: "Shell_weight", label: "Shell weight (0.01 - 1.0)", min: 0.01, max: 1.0 }
          ].map(({ field, label, min, max, placeholder }) => {
            const value = form[field];
            const num = parseFloat(value);
            const isOutOfRange = value !== "" && (num < min || num > max);

            return (
              <TextField
                key={field}
                label={label}
                name={field}
                value={value}
                onChange={handleChange}
                type="number"
                fullWidth
                margin="normal"
                inputProps={{ step: "0.001", min, max }}
                placeholder={placeholder}
                disabled={isLoading}
                error={isOutOfRange}
                helperText={
                  isOutOfRange
                    ? `Giá trị nên nằm trong khoảng ${min} - ${max}`
                    : ""
                }
              />
            );
          })}

          <div className="flex items-center w-full mt-3 mb-3">
            <button
              type="submit"
              disabled={!isFormValid || isLoading}
              className={`btn-org btn-lg w-full flex gap-3 items-center justify-center transition ${(!isFormValid || isLoading) ? "opacity-50 cursor-not-allowed" : "cursor-pointer"
                }`}
            >
              {isLoading ? <CircularProgress size={20} color="inherit" /> : <> <MdSend /> Dự đoán </>}
            </button>
          </div>
        </form>

        {result && (
          <Alert severity="success" sx={{ mt: 3 }}>
            <Typography><strong>{result.message}</strong></Typography>
            <Typography>Vòng tuổi (Rings): {result.predicted_rings}</Typography>
            <Typography>Tuổi ước tính: {result.estimated_age} năm</Typography>
          </Alert>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 3 }}>
            {error}
          </Alert>
        )}
      </Paper>
    </Container>
  );
}

export default App;
