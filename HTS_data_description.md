# Data Description

- **ID:** Household number + Person number  

- **sheet_code:** Household number  

- **relation:**  
  - 1: Head of Household  
  - 2: Spouse  
  - 3: Child  
  - 4: Parent  
  - 5: Others  

- **sex:**  
  - 1: Male  
  - 2: Female  

- **age_code:**  
  - **1:** Ages 0–11  
  - **2:** Ages 12–17  
  - **3:** Ages 18–22  
  - **4:** Ages 23–27  
  - **5:** Ages 28–32  
  - **6:** Ages 33–37  
  - **7:** Ages 38–42  
  - **8:** Ages 43–47  
  - **9:** Ages 48–52  
  - **10:** Ages 53–57  
  - **11:** Ages 58–62  
  - **12:** Ages 63–69  
  - **13:** Ages 70–999  

- **job_type:**  
  - 1: Student  
  - 2: Housewife/Unemployed (includes children not in school)  
  - 3: Professionals and Related Workers  
  - 4: Service Workers  
  - 5: Sales Workers  
  - 6: Managers and Office Workers  
  - 7: Skilled Workers in Agriculture, Forestry, and Fisheries  
  - 8: Technicians/Machine Operators/Short-Term Laborers  
  - 9: Others  

- **ID_trip_cnt:** Trip count for each ID starting from 1  

- **start_type:**  
  - 1: Home  
  - 2: Workplace  
  - 3: School  
  - 4: Others  
  - **start_zcode:**  
  Refer to the Excel file for details.

- **start_zcode_num:**  
  Numbering for "start_zcode" starting from 0, arranged in alphanumerical order.

- **start_time:**  
  Starting time in hours (e.g., 0.5 represents 30 minutes).

- **{start; end; trip}_time_{round; num}_{6;12}:**  
  Columns for start, end, or trip times, with two versions:
  - **round:** Time values rounded to the nearest 6 or 12 minutes.
  - **num:** Numerical representation based on 6- or 12-minute intervals.

- **start_time_num_{6; 12}:**  
  Numerical timestep based on a unit of either 6 or 12 minutes.

- **prev_act_{num: label}:**  
  Columns representing the previous action with both a number and a label.  
  The numbering starts from {home; 0}.

- **prev_mode_{num: label}:**  
  Columns representing the previous mode with both a number and a label.  
  The numbering starts from {stay; 0}.

- **{start; end}_{home; travel; work; education; academy; leisure; shopping; escort}_time:**  
  For each activity type (home, travel, work, education, academy, leisure, shopping, escort), these columns record either the first or last start/end time of the activity.

- **first_trip:**  
  Indicates the first trip.

- **last_trip:**  
  Indicates the last trip.