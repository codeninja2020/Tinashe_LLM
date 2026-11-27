// Public knowledge base for the Tinashe assistant.
// Curated from data/ (CV, LinkedIn export, bio) plus current roles.
// Third-party personal details (family, relationships) are deliberately left out
// because this is served on a public website.

export const KNOWLEDGE = `
# Tinashe Jambo — public profile

Contact: jambotinashe@gmail.com · LinkedIn: linkedin.com/in/tinashejambo · GitHub: github.com/codeninja2020
Based in Cape Town, South Africa. Originally from Rusape, Zimbabwe.

## Current roles
- Data engineer, AWS-centric: Glue, Lambda, S3, RDS/MSSQL, Redshift, Neptune, Terraform, GitHub Actions. Hands-on Snowflake (SQL connectors, virtual warehouse autoscaling).
- Co-founder and COO of Texcite (Pty) Ltd — smart water metering and IoT data analytics.
- Doctoral researcher (PhD candidate since Feb 2023) at the University of Cape Town, Computational and Continuum Mechanics research group: patient-specific cardiac mechanics using finite-element models, MRI segmentation (Simpleware, 3D Slicer), optimisation (multi-objective gradient descent, genetic algorithms), feed-forward and graph neural networks, HPC.
- Module Lead at NewBridge Graduate Institute (NGI): EEPC316 (PCB Design) and EEMS316 (Modelling & Simulation).
- Teaches the Vibrations module of CIV1011S Engineering Mechanics at UCT Civil Engineering.
- Ventures: MeshAI (current focus), Paysta, Texcite, Madam Waste, Yah Energy (renewables), ABCD Electricity (solar PV forecasting with satellite data, incubated at UCT GSB Solution Space).

## Current startups
- MeshAI (current work, meshai.texcite.co.za): AI for CAD and simulation. An add-in for Autodesk Fusion (macOS and Windows): describe a part in a sentence, or upload a sketch, photo, PDF, DXF or SVG, and it builds an editable parametric model in the Fusion timeline with key dimensions as user parameters. Modes: Solid, Sheet Metal, Surface, Sketch, Mesh and Assembly. It retries automatically when a step fails, and an early prototype records a workflow and reruns it with changes. Powered by Anthropic's Claude. Roadmap: extending past CAD into meshing and simulation, starting with GiD (STEP export, AI-proposed meshing commands the engineer approves), then mesh-quality validation and solver help; Onshape, SolidWorks, Ansys and ParaView connectors are planned, plus company design standards and design checks. 7-day free trial, then R99/month; pilots available on a company's own parts.
- Paysta: a USD-to-ZAR fintech payments app for moving and converting dollars into rand. Web app on Cloudflare with a Postgres backend, plus an iOS app.
- Built CarfinderAi (AI-enabled car marketplace iOS app) and an AI calling-agent MVP for a concierge company.
- ECSA Candidate Engineer.

## Education
- PhD candidate, Engineering — University of Cape Town (computational biomechanics).
- MSc Mechanical Engineering (Mechanics and Materials) — UCT. Eskom Power Plant Engineering Institute–funded project on the influence of post-weld heat treatment on the tensile behaviour of P91 weldments, using a Gleeble 3800, high-temperature tensile testing and Digital Image Correlation; SEM/EDS microstructural analysis.
- BSc Mechanical Engineering (Engineering Design) — UCT. Distinction thesis; top 5 final-year design projects; Dean's Merit List in first year.
- Schooling: Tasimukira Primary, Marist Brothers Nyanga, St. Ignatius College (A-Levels).

## Certifications
Dassault Systèmes Associate Mechanical Design (SolidWorks); Supervised, Unsupervised and Advanced Learning Algorithms (machine learning); IBM Data Analyst Professional Certificate; Mobile and web programming (Java, Swift/iOS, HTML, JavaScript, CSS) — HKUST; Multi-objective optimisation.

## Skills
Python (pandas, NumPy, scikit-learn, PyTorch, TensorFlow), SQL, BigQuery, IBM Db2, MongoDB, Firebase, Power BI, Looker Studio, Excel, MATLAB, ABAQUS FEA, SolidWorks, AutoCAD, Arduino and 3D printing, iOS/Android development (Swift, Java), Node.js, Angular, web front end, LLMs and RAG systems, project management (Gantt, scope, WBS). Engineering: solid mechanics, vibrations, fluid mechanics, thermodynamics/heat transfer/HVAC, mechanical design.

## Experience history
- 2023–present: Researcher, UCT Computational and Continuum Mechanics group. Built a RAG LLM for querying research papers, hosted on an Azure VM.
- 2020–present: Academic tutor at UCT (Stress Analysis MEC3076F, Mechanical Design & CAD MEC2048S, Engineering Drawing MEC1007F, 3rd-year Design MEC3082S). Jul–Dec 2023: Teaching Assistant / Assistant Lecturer for CIV1007S Structural Mechanics.
- Sep 2022 – Nov 2023: Business Development Officer, Madam Waste — biogas waste-to-energy consulting, reviewed UNIDO-funded biogas demonstration projects, wrote an investor guide, led RFP responses, built an automated scraper that emailed relevant UNIDO/GIZ RFPs daily, and set up a partnership to review WTE projects for a R150m investment fund.
- 2021–2022: Mechanical Engineer and Data Analyst, Texcite — supervised engineering interns (impact mole, ball thrower, 3-tonne ball mill, walk-through sanitiser), analysed music-streaming and social-media data for artists and tour planning (Python, SQL, dashboards), built a ticket payment portal integrated with mobile money.
- Mar–Oct 2021: Peer Mentor, Mastercard Foundation Scholars Program at UCT.
- Dec 2019: Intern, Gravitron — thermoelectric (Peltier) generator to power a mobile smoke machine.
- Jun–Sep 2019: Intern, UCT Future Water Institute — waterless, fertiliser-producing urinal with a pedal-driven mixing mechanism.
- 2016–2018: Mechanical Engineer, Metal Components Manufacturers (Harare) — mining-equipment drawings, gold-plant layouts, biscuit-machine manufacturing, maintenance inspections.
- 2014–2016: Innovation & Projects Engineer, Zimbabwe Newspapers Group (Zimpapers).
- 2014–2019: Founder, MICROBOTS (Divsol Pvt Ltd) — designed a Bluetooth vehicle immobiliser (Electronic Clamp), nominated at ITU Telecom World Durban 2018 representing Zimbabwe; built the Parkrite parking system (iOS, Android, Node.js) used in Zambia for 50,000+ transactions; exam-paper security solution with ZIMSEC covering 2,500 centres.

## Personal
Christian; enjoys Afrobeat and gospel music; favourite colour is blue. Member of the UCT Entrepreneurs Club. Writes about entrepreneurship and innovation on Medium. Interested in computational mechanics, machine learning, data, renewable energy and sustainability.
`;
