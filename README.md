# Accessible Route Planning System

## Project Overview

This system provides intelligent route planning specifically designed for differently abled and elderly citizens. It uses an enhanced A* algorithm with multi-criteria optimization to generate accessible, comfortable, and safe routes.

## Key Features

- **Personalized User Profiles**: Custom mobility constraints and preferences
- **Multi-Criteria Optimization**: Distance, surface quality, slope, barriers, energy consumption
- **Real-Time Updates**: Community feedback and obstacle reporting
- **Accessibility-First Design**: Wheelchair-friendly, elderly-accessible routes
- **Interactive Visualization**: Color-coded route segments with accessibility overlays

## System Architecture

### Core Components

1. **User Management System**
   - User registration and profile creation
   - Mobility aid type specification
   - Accessibility constraint configuration

2. **Map Data Processing**
   - OpenStreetMap integration for target regions
   - Accessibility attribute annotation
   - Real-time community feedback integration

3. **Enhanced A* Algorithm**
   - Multi-criteria cost function
   - Personalized constraint filtering
   - Dynamic route optimization

4. **Route Visualization**
   - Interactive map interface
   - Accessibility overlay system
   - Detailed navigation guidance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd smart-accessible-routing-a-star
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python manage.py db init
python manage.py db migrate
python manage.py db upgrade
```

6. Run the application:
```bash
python app.py
```

## Project Structure

```
smart-accessible-routing-a-star/
├── app/
│   ├── __init__.py
│   ├── models/
│   ├── routes/
│   ├── services/
│   └── utils/
├── data/
│   ├── maps/
│   └── user_profiles/
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
├── tests/
├── requirements.txt
├── app.py
└── README.md
```

## Usage

1. **User Registration**: Create an account and set up your mobility profile
2. **Route Planning**: Enter start and destination points
3. **Route Selection**: Choose from optimized routes based on your preferences
4. **Navigation**: Follow the detailed accessibility-aware guidance
5. **Feedback**: Provide feedback to improve the system

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue in the repository.