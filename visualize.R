
# =============================================================================
# Air Pollution Forecasting — R Visualization Suite
# Run: Rscript visualize.R   OR open in RStudio
#
# Install packages (run once):
# install.packages(c("ggplot2","dplyr","tidyr","leaflet",
#                    "viridis","plotly","htmlwidgets","patchwork",
#                    "scales","ggridges","ggcorrplot","RColorBrewer"))
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(leaflet)
library(viridis)
library(plotly)
library(htmlwidgets)
library(patchwork)      # combine ggplots with + / /
library(scales)
library(ggridges)
library(RColorBrewer)

# Paths
DATA_DIR <- "data/r_exports"
OUT_DIR  <- "data/r_figures"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Load all exported files
fc       <- read.csv(file.path(DATA_DIR, "forecast_series.csv"))
stations <- read.csv(file.path(DATA_DIR, "station_evaluation.csv"))
hist_df  <- read.csv(file.path(DATA_DIR, "training_history.csv"))
fc24     <- read.csv(file.path(DATA_DIR, "forecast_24h.csv"))
monthly  <- read.csv(file.path(DATA_DIR, "monthly_state_pm25.csv"))
diurnal  <- read.csv(file.path(DATA_DIR, "diurnal_pattern.csv"))
metrics  <- read.csv(file.path(DATA_DIR, "model_metrics.csv"))

EPA_STD <- 35   # µg/m³

# ── Theme ─────────────────────────────────────────────────────────────────────
theme_air <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey50", size = 11),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom"
  )


# =============================================================================
# R PLOT 1: Forecast vs Actual — ggplot2
# =============================================================================
fc_week <- fc[1:336, ]   # 2-week window

p_forecast <- ggplot(fc_week, aes(x = hour)) +
  geom_ribbon(aes(ymin = pmin(actual, predicted),
                  ymax = pmax(actual, predicted)),
              fill = "#9C27B0", alpha = 0.15) +
  geom_line(aes(y = actual,    color = "Actual"),    linewidth = 0.9, alpha = 0.9) +
  geom_line(aes(y = predicted, color = "Predicted"), linewidth = 0.9,
            linetype = "dashed", alpha = 0.9) +
  geom_hline(yintercept = EPA_STD, color = "#D32F2F", linetype = "dotted",
             linewidth = 0.9) +
  annotate("text", x = 20, y = EPA_STD + 1.5,
           label = "EPA 24h Standard (35 µg/m³)",
           color = "#D32F2F", size = 3.5) +
  scale_color_manual(
    values = c("Actual" = "#1565C0", "Predicted" = "#E53935"),
    name   = NULL
  ) +
  labs(
    title    = "1-Hour Ahead PM2.5 Forecast — STGNN-LSTM",
    subtitle = sprintf("MAE = %.3f µg/m³  |  R² = %.3f  |  MAPE = %.1f%%",
                       metrics$mae, metrics$r2, metrics$mape),
    x = "Hour Index",
    y = expression(PM[2.5] ~ (µg/m^3))
  ) +
  theme_air

ggsave(file.path(OUT_DIR, "R1_forecast_vs_actual.png"),
       p_forecast, width = 14, height = 5, dpi = 200)
cat("✅ R Plot 1 saved\n")


# =============================================================================
# R PLOT 2: Training Loss Curves — ggplot2
# =============================================================================
hist_long <- hist_df %>%
  pivot_longer(c(train_loss, val_loss), names_to = "split", values_to = "loss") %>%
  mutate(split = recode(split, train_loss = "Train", val_loss = "Validation"))

p_loss <- ggplot(hist_long, aes(x = epoch, y = loss, color = split)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 1.5, alpha = 0.6) +
  geom_smooth(se = FALSE, method = "loess", linewidth = 0.6, linetype = "dashed", alpha = 0.5) +
  scale_color_manual(values = c("Train" = "#1565C0", "Validation" = "#E53935"),
                     name = NULL) +
  labs(
    title    = "STGNN-LSTM Training Curves",
    subtitle = "Huber Loss (delta=0.5) with early stopping",
    x = "Epoch", y = "Huber Loss"
  ) +
  theme_air

ggsave(file.path(OUT_DIR, "R2_training_curves.png"),
       p_loss, width = 10, height = 5, dpi = 200)
cat("✅ R Plot 2 saved\n")


# =============================================================================
# R PLOT 3: Residuals Analysis — 4-panel
# =============================================================================
# 3a: Residual histogram
p3a <- ggplot(fc, aes(x = residual)) +
  geom_histogram(bins = 60, fill = "#1565C0", color = "white", alpha = 0.8) +
  geom_vline(xintercept = 0,           color = "#D32F2F", linewidth = 1.2, linetype = "dashed") +
  geom_vline(xintercept = mean(fc$residual), color = "#FF6F00", linewidth = 1,
             linetype = "dashed") +
  labs(title = "Residual Distribution",
       x = "Residual (predicted − actual, µg/m³)", y = "Count") +
  theme_air

# 3b: Residual vs actual (heteroscedasticity check)
p3b <- ggplot(fc, aes(x = actual, y = residual)) +
  geom_hex(bins = 50, aes(fill = after_stat(count))) +
  scale_fill_viridis(option = "plasma", name = "Count") +
  geom_hline(yintercept = 0, color = "red", linewidth = 1, linetype = "dashed") +
  geom_smooth(method = "loess", se = TRUE, color = "white", linewidth = 0.8) +
  labs(title = "Residuals vs Actual",
       x = expression(Actual ~ PM[2.5]), y = "Residual") +
  theme_air + theme(legend.position = "right")

# 3c: Predicted vs Actual hex
p3c <- ggplot(fc, aes(x = actual, y = predicted)) +
  geom_hex(bins = 60, aes(fill = after_stat(count))) +
  scale_fill_viridis(option = "magma", name = "Count") +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1.2, linetype = "dashed") +
  labs(title = expression(bold("Predicted vs Actual PM"[2.5])),
       x = expression(Actual ~ (µg/m^3)),
       y = expression(Predicted ~ (µg/m^3))) +
  theme_air + theme(legend.position = "right")

# 3d: Model vs baseline comparison
comp_df <- data.frame(
  model  = c("STGNN-LSTM", "STGNN-LSTM", "Persistence", "Persistence"),
  metric = c("MAE", "RMSE", "MAE", "RMSE"),
  value  = c(metrics$mae, metrics$rmse, metrics$persist_mae, metrics$persist_rmse)
)
p3d <- ggplot(comp_df, aes(x = metric, y = value, fill = model)) +
  geom_col(position = "dodge", width = 0.6, color = "white") +
  geom_text(aes(label = round(value, 2)), position = position_dodge(0.6),
            vjust = -0.4, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("STGNN-LSTM" = "#1565C0", "Persistence" = "#E53935"),
                    name = NULL) +
  labs(title = "Model vs Persistence Baseline",
       subtitle = sprintf("MAE improvement: %.1f%%", metrics$improvement_pct),
       x = "Metric", y = expression(Error ~ (µg/m^3))) +
  theme_air

p_residuals <- (p3a | p3b) / (p3c | p3d) +
  plot_annotation(title = "STGNN-LSTM — Error & Residual Analysis",
                  theme = theme(plot.title = element_text(face="bold", size=16)))

ggsave(file.path(OUT_DIR, "R3_residual_analysis.png"),
       p_residuals, width = 16, height = 12, dpi = 200)
cat("✅ R Plot 3 saved\n")


# =============================================================================
# R PLOT 4: 24-Hour Ahead Forecast with Uncertainty Band
# =============================================================================
p_fc24 <- ggplot(fc24, aes(x = hour_ahead)) +
  geom_ribbon(aes(ymin = mean_pm25 - std_pm25,
                  ymax = mean_pm25 + std_pm25),
              fill = "#FF6F00", alpha = 0.25) +
  geom_ribbon(aes(ymin = mean_pm25 - 0.5*std_pm25,
                  ymax = mean_pm25 + 0.5*std_pm25),
              fill = "#FF6F00", alpha = 0.35) +
  geom_line(aes(y = mean_pm25), color = "#E65100", linewidth = 1.8) +
  geom_point(aes(y = mean_pm25), color = "#E65100", size = 3) +
  geom_hline(yintercept = EPA_STD, color = "#D32F2F", linewidth = 1,
             linetype = "dashed") +
  annotate("text", x = 2, y = EPA_STD + 1,
           label = "EPA Standard (35 µg/m³)",
           color = "#D32F2F", size = 3.5, hjust = 0) +
  scale_x_continuous(breaks = seq(1, 24, 2),
                     labels = paste0("+", seq(1, 24, 2), "h")) +
  labs(
    title    = "24-Hour Ahead PM2.5 Forecast — Recursive Prediction",
    subtitle = "National average across all stations  |  Shaded bands: ±0.5σ and ±1σ",
    x        = "Hours Ahead",
    y        = expression(PM[2.5] ~ (µg/m^3))
  ) +
  theme_air

ggsave(file.path(OUT_DIR, "R4_24h_forecast.png"),
       p_fc24, width = 13, height = 6, dpi = 200)
cat("✅ R Plot 4 saved\n")


# =============================================================================
# R PLOT 5: Seasonal + Diurnal Patterns
# =============================================================================
month_labels <- c("Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec")

# 5a: Monthly PM2.5 ridge plot by state (top 15 states by data volume)
top_states <- monthly %>%
  count(state_name, sort = TRUE) %>%
  head(12) %>%
  pull(state_name)

p5a <- monthly %>%
  filter(state_name %in% top_states) %>%
  mutate(month_f = factor(month, levels = 1:12, labels = month_labels)) %>%
  ggplot(aes(x = value, y = reorder(state_name, value, median),
             fill = after_stat(x))) +
  ggridges::geom_density_ridges_gradient(scale = 1.5, rel_min_height = 0.01) +
  scale_fill_viridis(option = "inferno", name = "PM2.5") +
  geom_vline(xintercept = EPA_STD, color = "white", linewidth = 0.8, linetype = "dashed") +
  labs(title = "PM2.5 Distribution by State — Ridge Plot",
       subtitle = "Top 12 states by data volume | White dashed = EPA standard",
       x = expression(PM[2.5] ~ (µg/m^3)), y = "State") +
  theme_air + theme(legend.position = "right")

# 5b: Diurnal pattern
p5b <- ggplot(diurnal, aes(x = hour)) +
  geom_ribbon(aes(ymin = mean - std, ymax = mean + std),
              fill = "#1565C0", alpha = 0.2) +
  geom_line(aes(y = mean),   color = "#1565C0", linewidth = 1.8) +
  geom_line(aes(y = median), color = "#FF6F00", linewidth = 1.2, linetype = "dashed") +
  geom_point(aes(y = mean),  color = "#1565C0", size = 2.5) +
  annotate("rect", xmin = 6, xmax = 9, ymin = -Inf, ymax = Inf,
           fill = "yellow", alpha = 0.15) +
  annotate("rect", xmin = 17, xmax = 20, ymin = -Inf, ymax = Inf,
           fill = "orange", alpha = 0.12) +
  annotate("text", x = 7.5, y = Inf, vjust = 1.5, size = 3,
           label = "Morning\nRush") +
  annotate("text", x = 18.5, y = Inf, vjust = 1.5, size = 3,
           label = "Evening\nRush") +
  scale_x_continuous(breaks = seq(0,23,3),
                     labels = paste0(seq(0,23,3), ":00")) +
  labs(title = "Diurnal PM2.5 Pattern",
       subtitle = "Mean (blue) and median (orange dashed)  |  Shaded: ±1σ",
       x = "Hour of Day",
       y = expression(PM[2.5] ~ (µg/m^3))) +
  theme_air

p_patterns <- p5a / p5b +
  plot_annotation(title = "Seasonal & Diurnal PM2.5 Patterns — 2022–2023",
                  theme = theme(plot.title = element_text(face="bold", size=16)))

ggsave(file.path(OUT_DIR, "R5_seasonal_diurnal.png"),
       p_patterns, width = 14, height = 13, dpi = 200)
cat("✅ R Plot 5 saved\n")


# =============================================================================
# R PLOT 6: State-Level Bar Chart (Top 20 worst states)
# =============================================================================
state_summary <- monthly %>%
  group_by(state_name) %>%
  summarise(mean_pm25 = mean(value, na.rm=TRUE),
            sd_pm25   = sd(value, na.rm=TRUE)) %>%
  arrange(desc(mean_pm25)) %>%
  head(20)

p_states <- ggplot(state_summary,
                   aes(x = reorder(state_name, mean_pm25), y = mean_pm25,
                       fill = mean_pm25)) +
  geom_col(color = "white", linewidth = 0.4) +
  geom_errorbar(aes(ymin = mean_pm25 - sd_pm25/2,
                    ymax = mean_pm25 + sd_pm25/2),
                width = 0.4, color = "grey40") +
  geom_hline(yintercept = 12, color = "#FF6F00", linewidth = 1,
             linetype = "dashed") +
  geom_hline(yintercept = 9, color = "#388E3C", linewidth = 1,
             linetype = "dashed") +
  annotate("text", x = 1.5, y = 12.5, label = "NAAQS Annual (12 µg/m³)",
           color = "#FF6F00", size = 3, hjust = 0) +
  annotate("text", x = 1.5, y = 9.5, label = "WHO Guideline (9 µg/m³)",
           color = "#388E3C", size = 3, hjust = 0) +
  scale_fill_gradient2(low = "#388E3C", mid = "#FFC107", high = "#D32F2F",
                       midpoint = 10, guide = "none") +
  coord_flip() +
  labs(title = "Top 20 States — Mean Annual PM2.5 (2022–2023)",
       subtitle = "Error bars = ±0.5 SD  |  Horizontal lines = regulatory standards",
       x = NULL, y = expression(Mean ~ PM[2.5] ~ (µg/m^3))) +
  theme_air

ggsave(file.path(OUT_DIR, "R6_state_pm25.png"),
       p_states, width = 11, height = 8, dpi = 200)
cat("✅ R Plot 6 saved\n")


# =============================================================================
# R PLOT 7: Interactive Leaflet Map — Station Forecast Error
# =============================================================================
stations_clean <- stations %>%
  filter(!is.na(latitude), !is.na(longitude), !is.na(mae),
         longitude > -130, longitude < -65,
         latitude  >   24, latitude  <  50)

pal_mae <- colorNumeric(
  palette = c("#388E3C", "#FFC107", "#D32F2F"),
  domain   = stations_clean$mae,
  na.color = "#BDBDBD"
)

map_mae <- leaflet(stations_clean) %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  addCircleMarkers(
    lng         = ~longitude,
    lat         = ~latitude,
    radius      = ~scales::rescale(mae, to = c(4, 12)),
    color       = ~pal_mae(mae),
    stroke      = TRUE, weight = 1.2, opacity = 0.9,
    fillOpacity = 0.85,
    popup = ~paste0(
      "<b>Station:</b> ", station_id, "<br>",
      "<b>State:</b> ",   state_name,  "<br>",
      "<b>MAE:</b> ",     round(mae, 3), " µg/m³<br>",
      "<b>RMSE:</b> ",    round(rmse, 3), " µg/m³"
    ),
    label = ~paste0(station_id, " | MAE=" , round(mae,2))
  ) %>%
  addLegend(
    position = "bottomright",
    pal    = pal_mae, values = ~mae,
    title  = "Forecast MAE (µg/m³)",
    opacity = 0.9
  ) %>%
  addControl(
    html = paste0(
      "<div style='background:white;padding:8px;border-radius:6px;font-size:13px;'>",
      "<b>STGNN-LSTM Forecast Error</b><br>",
      "National MAE: ", round(mean(stations_clean$mae, na.rm=TRUE), 3), " µg/m³",
      "</div>"
    ),
    position = "topright"
  )

saveWidget(map_mae,
           file.path(OUT_DIR, "R7_station_mae_map.html"),
           selfcontained = TRUE)
cat("✅ R Plot 7 (interactive map) saved\n")


# =============================================================================
# R PLOT 8: Interactive Plotly — Monthly PM2.5 Trend
# =============================================================================
monthly_nat <- monthly %>%
  mutate(date = as.Date(paste(year, month, "01", sep="-"))) %>%
  group_by(date) %>%
  summarise(mean_pm25 = mean(value, na.rm=TRUE),
            sd_pm25   = sd(value, na.rm=TRUE))

p_trend <- plot_ly(monthly_nat) %>%
  add_ribbons(
    x = ~date,
    ymin = ~mean_pm25 - sd_pm25,
    ymax = ~mean_pm25 + sd_pm25,
    fillcolor = "rgba(33,150,243,0.2)",
    line = list(color = "transparent"),
    name = "±1 SD"
  ) %>%
  add_lines(
    x = ~date, y = ~mean_pm25,
    line = list(color = "#1565C0", width = 2.5),
    name = "National Mean PM2.5"
  ) %>%
  add_lines(
    x = range(monthly_nat$date),
    y = c(EPA_STD, EPA_STD),
    line = list(color = "red", dash = "dot", width = 1.5),
    name = "EPA Standard"
  ) %>%
  layout(
    title  = list(text = "National Monthly PM2.5 Trend 2022–2023", font = list(size=16)),
    xaxis  = list(title = "Date"),
    yaxis  = list(title = "PM2.5 (µg/m³)"),
    legend = list(orientation = "h", x = 0, y = -0.15),
    hovermode = "x unified"
  )

saveWidget(p_trend,
           file.path(OUT_DIR, "R8_monthly_trend_plotly.html"),
           selfcontained = TRUE)
cat("✅ R Plot 8 (interactive Plotly trend) saved\n")


# =============================================================================
# Summary
# =============================================================================
cat("\n", strrep("=", 55), "\n")
cat("  ✅  ALL R VISUALIZATIONS SAVED\n")
cat(strrep("=", 55), "\n")
cat("  R1  Forecast vs Actual          → R1_forecast_vs_actual.png\n")
cat("  R2  Training Loss Curves        → R2_training_curves.png\n")
cat("  R3  Residual Analysis (4-panel) → R3_residual_analysis.png\n")
cat("  R4  24h Forecast + Uncertainty  → R4_24h_forecast.png\n")
cat("  R5  Seasonal & Diurnal Patterns → R5_seasonal_diurnal.png\n")
cat("  R6  State-Level PM2.5 Bar Chart → R6_state_pm25.png\n")
cat("  R7  Interactive Leaflet Map     → R7_station_mae_map.html\n")
cat("  R8  Interactive Plotly Trend    → R8_monthly_trend_plotly.html\n")
cat(strrep("=", 55), "\n")
