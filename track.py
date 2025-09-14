import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from ultralytics import YOLO
from tracker import Tracker
from datetime import datetime


def generate_environmental_report(video_path, final_count_path):
    os.makedirs("results", exist_ok=True)
    
    with open(final_count_path, 'r') as f:
        count_data = int(f.read().split(":")[-1].strip())
    
    CO2_PER_TREE = 22  # kg CO₂ absorbed per year
    O2_PER_TREE = 118  # kg O₂ produced per year
    
    total_co2 = count_data * CO2_PER_TREE
    total_o2 = count_data * O2_PER_TREE
    
    # Generate Bar Chart
    plt.figure(figsize=(6, 4))
    categories = ["Trees Count", "CO2 Sequestered (kg)", "O2 Produced (kg)"]
    values = [count_data, total_co2, total_o2]
    plt.bar(categories, values, color=['green', 'blue', 'orange'])
    plt.ylabel("Amount")
    plt.title("Environmental Impact Summary")
    chart_path = "results/environmental_impact.png"
    plt.savefig(chart_path)
    plt.close()
    
    # Generate PDF Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"results/environmental_report_{timestamp}.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Tree Detection & Environmental Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, f"Total Trees Detected: {count_data}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Estimated CO2 Sequestration: {total_co2} kg/year", ln=True)
    pdf.cell(200, 10, f"Estimated O2 Production: {total_o2} kg/year", ln=True)
    pdf.ln(10)
    
    if os.path.exists(chart_path):
        pdf.image(chart_path, x=40, w=120)
    
    pdf.output(report_path)
    return report_path


def run_tracking(start_x, start_y, end_x, end_y, video_path):
    os.makedirs("results", exist_ok=True)
    model = YOLO('bestk.pt')
    class_list = ['tree']
    tracker = Tracker(start_y)
    counter_down = set()
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_video_path = os.path.join("results", f"processed_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame)
        detections = results[0].boxes.data.detach().cpu().numpy() if results[0].boxes.data is not None else []
        
        detected_objects = []
        if len(detections) > 0:
            px = pd.DataFrame(detections, columns=["x1", "y1", "x2", "y2", "score", "class_id"])
            for _, row in px.iterrows():
                x1, y1, x2, y2, _, class_id = map(int, row)
                if class_id == 0:  # Ensure the detected object is a tree
                    detected_objects.append([x1, y1, x2, y2])
        
        bbox_id = tracker.update(detected_objects)
        for x1, y1, x2, y2, obj_id in bbox_id:
            if y1 <= start_y <= y2:
                counter_down.add(obj_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.line(frame, (0, start_y), (width - 1, start_y), (0, 0, 255), 3)
        cv2.putText(frame, f'count: {len(counter_down)}', (60, 40), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 255), 2)
        
        out.write(frame)
        cv2.imshow("Processed Video", frame)# new add

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    final_count_path = os.path.join("results", f"final_count_{os.path.basename(video_path)}.txt")
    with open(final_count_path, 'w') as f:
        f.write(f"Total trees counted: {len(counter_down)}")
    
    report_path = generate_environmental_report(output_video_path, final_count_path)
    return output_video_path, final_count_path, report_path

