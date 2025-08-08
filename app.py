from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import sqlite3
from sqlite3 import Error
import re
import os
import threading
from functools import wraps
from difflib import get_close_matches
from werkzeug.security import generate_password_hash, check_password_hash
from flask import redirect, url_for, flash
from forms import LoginForm, ChangePasswordForm, RegisterForm
import plotly.graph_objects as go
from plotly.io import to_html
import time

# Database configuration
DATABASE_NAME = "chatbot.db"
DATABASE_LOCK = threading.Lock()

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
app.config['WTF_CSRF_SECRET_KEY'] = os.urandom(24).hex()

# Conversation states
STATE_NORMAL = 'normal'
STATE_ADD_ITEM_SALES = 'add_item_sales'
STATE_ADD_ITEM_PRICE = 'add_item_price'
STATE_ADD_ITEM_COST = 'add_item_cost'
STATE_SELECT_METRIC = 'select_metric'
STATE_SELECT_CHART_TYPE = 'select_chart_type'
STATE_SELECT_FILTER = 'select_filter'

def get_db():
    db = Database()
    return db.get_connection()

# Database connection manager
class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False, timeout=10)
            cls._instance.conn.execute('PRAGMA journal_mode=WAL')
        return cls._instance

    def get_connection(self):
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()

def with_db_connection(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with DATABASE_LOCK:
            db = Database()
            conn = db.get_connection()
            try:
                result = func(conn, *args, **kwargs)
                conn.commit()
                return result
            except Exception as e:
                print(f"DB error: {e}")
                conn.rollback()
                raise
    return wrapper

def prepare_chart_data(chart_type, data_type, filter_option=None, limit=None):
    conn = get_db()
    cursor = conn.cursor()
    
    # Correct column mapping
    value_column = {
        "sales": "item_sales",
        "price": "item_price",
        "cost": "item_cost",
        "profit": "(item_price - item_cost)"
    }.get(data_type, "item_sales")
    
    # Base query with correct table
    query = f"""
        SELECT item_name, {value_column} AS value FROM items
    """
    
    filters = []
    if filter_option == "top5":
        filters.append("ORDER BY value DESC LIMIT 5")
    elif filter_option == "low_stock":
        query = "SELECT item_name, item_sales AS value FROM items ORDER BY item_sales ASC"
        if limit:
            filters.append(f"LIMIT {limit}")
    elif filter_option == "most_profitable":
        query = "SELECT item_name, (item_price - item_cost) AS value FROM items ORDER BY value DESC"
        if limit:
            filters.append(f"LIMIT {limit}")
    elif filter_option == "affordable":
        query = "SELECT item_name, item_price AS value FROM items ORDER BY item_price ASC"
        if limit:
            filters.append(f"LIMIT {limit}")
    elif filter_option == "low_cost":
        query = "SELECT item_name, item_cost AS value FROM items ORDER BY item_cost ASC"
        if limit:
            filters.append(f"LIMIT {limit}")
    elif limit:
        filters.append(f"LIMIT {limit}")
        
    # Final query construction
    query += " " + " ".join(filters)
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    labels = [row[0] for row in rows]
    values = [float(row[1]) for row in rows]
    unit = "$" if data_type in ["price", "cost", "profit"] else ""
    title = f"{data_type.capitalize()} by Product"
    
    return {
        "type": chart_type,
        "labels": labels,
        "values": values,
        "title": title,
        "unit": unit
    }

def generate_plotly_chart(chart_config):
    print(f"Generating chart with config: {chart_config}")
    labels = chart_config['labels']
    values = chart_config['values']
    title = chart_config['title']
    chart_type = chart_config['type']
    unit = chart_config.get('unit', '')
    
    # Check if we have data
    if not labels or not values:
        return "<div class='graph-container'><p>No data available for this chart.</p></div>"
    
    # Add dollar signs if it's a monetary value
    if unit == '$':
        text = [f"${v:,.2f}" for v in values]
    else:
        text = [f"{v:,.0f}" for v in values]

    if chart_type == 'bar':
        fig = go.Figure([go.Bar(
            x=labels, 
            y=values,
            text=text,
            textposition='auto',
            marker_color='#4a6cf7'
        )])
    elif chart_type == 'line':
        fig = go.Figure([go.Scatter(
            x=labels, 
            y=values,
            mode='lines+markers+text',
            text=text,
            textposition='top center',
            line=dict(color='#4a6cf7', width=3)
        )])
    elif chart_type == 'pie':
        fig = go.Figure([go.Pie(
            labels=labels, 
            values=values,
            textinfo='label+percent',
            texttemplate='%{label}<br>%{percent}<br>(%{value:,.0f})',
            marker=dict(colors=['#4a6cf7', '#3a5bef', '#2a4bdf', '#1a3bcf', '#0a2bbf'])
        )])
    elif chart_type == 'doughnut':
        fig = go.Figure([go.Pie(
            labels=labels, 
            values=values,
            hole=0.5,
            textinfo='label+percent',
            texttemplate='%{label}<br>%{percent}<br>(%{value:,.0f})'
        )])
    else:
        return "<div class='graph-container'><p>Invalid chart type selected</p></div>"
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        width=550,
        height=400,
        margin=dict(t=60, l=50, r=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        showlegend=True if chart_type in ['pie', 'doughnut'] else False
    )
    
    # Generate a unique div ID
    div_id = f"plotly-chart-{int(time.time() * 1000)}"
    
    # Convert figure to JSON for JavaScript
    fig_json = fig.to_json()
    
    # Create the HTML with embedded JavaScript that will create the plot
    html_output = f"""
    <div class="graph-container">
        <div id="{div_id}" style="width: 100%; height: 400px;"></div>
        <script>
            (function() {{
                var plotData = {fig_json};
                var config = {{
                    'displayModeBar': true,
                    'displaylogo': false,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'responsive': true
                }};
                
                // Wait for Plotly to be available
                function createPlot() {{
                    if (typeof Plotly !== 'undefined') {{
                        console.log('Creating plot for {div_id}');
                        Plotly.newPlot('{div_id}', plotData.data, plotData.layout, config);
                    }} else {{
                        console.log('Plotly not ready, retrying...');
                        setTimeout(createPlot, 100);
                    }}
                }}
                
                // Start trying to create the plot
                createPlot();
            }})();
        </script>
    </div>
    """
    
    print(f"Generated HTML snippet (first 200 chars): {html_output[:200]}...")
    return html_output

def admin_required(func):
    """
    Decorator to require admin role for accessing certain routes.
    Redirects non-admin users to index with error message.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if user is logged in
        if 'username' not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for('login'))
        
        # Check if user has admin role
        if session.get('role') != 'admin':
            flash("Admin access required. You do not have permission to view this page.", "error")
            return redirect(url_for('index'))
            
        # User is admin, proceed with the original function
        return func(*args, **kwargs)
    return wrapper

@with_db_connection
def initialize_database(conn):
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL UNIQUE,
            item_sales INTEGER DEFAULT 0,
            item_price REAL NOT NULL,
            item_cost REAL NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT NOT NULL,
            generated_sql TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("SELECT COUNT(*) FROM items")
    if cursor.fetchone()[0] == 0:
        sample_items = [
            ("Notebook", 150, 5.99, 2.50),
            ("Pen", 300, 1.99, 0.75),
            ("Pencil", 200, 0.99, 0.25),
            ("Eraser", 100, 1.49, 0.50),
            ("Ruler", 75, 2.99, 1.00)
        ]
        cursor.executemany("""
            INSERT INTO items (item_name, item_sales, item_price, item_cost)
            VALUES (?, ?, ?, ?)
        """, sample_items)
        print("Added sample items.")
    
    # Add users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'guest' -- 'admin' or 'guest'
        )
    """)
    
    # Create default admin if none exists (change credentials!)
    cursor.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    if cursor.fetchone()[0] == 0:
        from werkzeug.security import generate_password_hash
        default_admin = {
            'username': 'admin',
            'password': 'ChangeThisPassword!123',
            'role': 'admin'
        }
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (default_admin['username'], generate_password_hash(default_admin['password']), default_admin['role'])
        )
        conn.commit()

@with_db_connection
def log_query(conn, user_query, generated_sql, bot_response):
    if user_query and generated_sql and bot_response:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_logs (user_query, generated_sql, bot_response)
            VALUES (?, ?, ?)
        """, (user_query, generated_sql, bot_response))

def clean_input(text):
    """Normalize input and handle common misspellings"""
    text = text.lower().strip()
    common_mistakes = {
        'pne': 'pen',
        'noebook': 'notebook',
        'rular': 'ruler',
        'lapop': 'laptop',
        'apne': 'pen',
        'notbok': 'notebook',
        'pencel': 'pencil'
    }
    return common_mistakes.get(text, text)

def find_closest_item(conn, user_input):
    """Find closest matching item using existing connection"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT item_name, item_sales FROM items")
        inventory_items = [(row[0], row[1]) for row in cursor.fetchall()]
        
        user_input = clean_input(user_input)
        
        # First try exact match
        for item, _ in inventory_items:
            if user_input == item.lower():
                return item
                
        # Get all item names for fuzzy matching
        item_names = [item[0] for item in inventory_items]
        matches = get_close_matches(user_input, [name.lower() for name in item_names], n=3, cutoff=0.6)
        
        if not matches:
            return None
            
        if len(matches) > 1:
            similar_start = [m for m in matches if m.startswith(user_input[:2])]
            if similar_start:
                matches = similar_start
                
        matched_item = next((item for item, _ in inventory_items if item.lower() == matches[0]), None)
        return matched_item
        
    except Error as e:
        print(f"Error in find_closest_item: {e}")
        return None

@with_db_connection
def get_item_info(conn, item_name, info_type):
    """Get item info using single database connection"""
    try:
        # First try exact match
        cursor = conn.cursor()
        exact_sql = f"""
            SELECT item_name, {info_type} FROM items
            WHERE LOWER(item_name) = LOWER(?)
            LIMIT 1
        """
        cursor.execute(exact_sql, (item_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0], result[1], exact_sql, False
            
        # Try fuzzy matching
        closest_match = find_closest_item(conn, item_name)
        if closest_match:
            cursor.execute(exact_sql, (closest_match,))
            result = cursor.fetchone()
            if result:
                return result[0], result[1], exact_sql, True
                
        # Fall back to partial matching
        fuzzy_sql = f"""
            SELECT item_name, {info_type} FROM items
            WHERE LOWER(item_name) LIKE LOWER(?)
            ORDER BY CASE 
                WHEN item_name LIKE ? THEN 0
                WHEN item_name LIKE ? THEN 1
                ELSE 2
            END
            LIMIT 1
        """
        cursor.execute(fuzzy_sql, (
            f"%{item_name}%",
            f"{item_name}%",
            f"%{item_name}%"
        ))
        result = cursor.fetchone()
        
        if result:
            return result[0], result[1], fuzzy_sql, False
            
        return None, None, None, False
        
    except Error as e:
        print(f"Error in get_item_info: {e}")
        return None, None, None, False

@with_db_connection
def add_new_item(conn, item_data):
    cursor = conn.cursor()
    cursor.execute("SELECT item_name FROM items WHERE LOWER(item_name) = LOWER(?)", (item_data['name'],))
    if cursor.fetchone():
        return False, "Item already exists"
    
    cursor.execute("""
        INSERT INTO items (item_name, item_sales, item_price, item_cost)
        VALUES (?, ?, ?, ?)
    """, (item_data['name'], item_data['sales'], item_data['price'], item_data['cost']))
    
    return True, f"New item added: {item_data['name']} (Sales: {item_data['sales']}, Price: ${item_data['price']:.2f}, Cost: ${item_data['cost']:.2f})"

def detect_query_type(user_message):
    """Detects the type of query and extracts relevant information, including graph requests"""
    user_message = user_message.lower().strip()
    query_patterns = {
        'price': [
            r'(?:price|how much)\s+(?:of|for|is)?\s*(?:an?|the)?\s*(.+)',
            r'what(?:\'s| is) the price\s+(?:of|for)?\s*(?:an?|the)?\s*(.+)'
        ],
        'sales': [
            r'sales\s+(?:of|for)?\s*(?:an?|the)?\s*(.+)',
            r'how many\s+(.+)\s+(?:have|were) sold',
            r'quantity sold\s+(?:of|for)?\s*(?:an?|the)?\s*(.+)'
        ],
        'cost': [
            r'cost\s+(?:of|to make)?\s*(?:an?|the)?\s*(.+)',
            r'production cost\s+(?:of)?\s*(?:an?|the)?\s*(.+)'
        ],
        'graph': [
            # Graph request patterns
            r'(?:show|display|create|generate).*(?:graph|chart|visual(?:ization|ize))\s*(?:of|for)?\s*(.+)',
            r'visualize\s+(?:the)?\s*(.+)',
            r'compare\s+(.+)',
            r'graph\s+(?:of|for)?\s*(.+)',
            r'chart\s+(?:of|for)?\s*(.+)'
        ]
    }
    
    for query_type, patterns in query_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, user_message)
            if match:
                item_name = next((g for g in match.groups() if g), None)
                if item_name:
                    # For graph requests, clean the extracted text
                    if query_type == 'graph':
                        item_name = re.sub(r'(graph|chart|visualize|compare|of|for)', '', item_name, flags=re.IGNORECASE).strip()
                    return query_type, re.sub(r'[\?\.!]*$', '', item_name.strip())
    
    return None, None

def detect_add_item_request(user_message):
    patterns = [
        r'add (?:a|an|the)?\s*(.+)',
        r'new item (.+)',
        r'create (?:a|an)?\s*entry for (.+)',
        r'i want to add (?:a|an)?\s*(.+)',
        r'i\'d like to add (?:a|an)?\s*(.+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_message.lower().strip())
        if match:
            return match.group(1).capitalize()
    return None

def format_response(query_type, requested_name, db_item_name, value, is_potential_typo=False):
    if value is None:
        conn = Database().get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT item_name FROM items")
            available_items = [row[0] for row in cursor.fetchall()]
            return f"I couldn't find '{requested_name}'. Available items: {', '.join(available_items)}"
        except Error as e:
            print(f"Error getting available items: {e}")
            return f"I couldn't find '{requested_name}'."
    
    if is_potential_typo:
        responses = {
            'price': f"Did you mean '{db_item_name}'? The price is ${value:.2f}",
            'sales': f"Did you mean '{db_item_name}'? We've sold {value} units",
            'cost': f"Did you mean '{db_item_name}'? The production cost is ${value:.2f}"
        }
    else:
        responses = {
            'price': f"The price of {db_item_name} is ${value:.2f}",
            'sales': f"We've sold {value} units of {db_item_name}",
            'cost': f"The production cost of {db_item_name} is ${value:.2f}"
        }
    
    return responses.get(query_type, f"{db_item_name}: {value}")

def detect_graph_request(user_message):
    """Detect if user wants to create a graph and extract subject"""
    patterns = [
        r'(?:show|display|create|generate).*(?:graph|chart|visual(?:ization|ize))',
        r'visualize\s+(?:the)?',
        r'compare\s+'
    ]
    
    for pattern in patterns:
        if re.search(pattern, user_message.lower()):
            return True
    return False

def get_graph_metric_options():
    """Return available graph options with clear instructions"""
    return {
        'response': "I can create graphs for:\n"
                   "1. Sales by product\n"
                   "2. Prices comparison\n"
                   "3. Cost breakdown\n"
                   "4. Profit analysis\n"
                   "Which would you like? (Enter 1-4 or 'cancel' to stop)",
        'options': ['sales', 'price', 'cost', 'profit']
    }

def get_chart_type_options():
    """Return available chart type options"""
    return {
        'response': "What type of chart would you like?\n"
                   "1. Bar chart\n"
                   "2. Line chart\n"
                   "3. Pie chart\n"
                   "4. Doughnut chart\n"
                   "Please choose (1-4):",
        'options': ['bar', 'line', 'pie', 'doughnut']
    }

def get_filter_options():
    """Return available filter options"""
    return {
        'response': "How would you like to filter the data?\n"
                   "1. All items\n"
                   "2. Top 5 items\n"
                   "3. Lowest cost items\n"
                   "4. Most profitable items\n"
                   "5. Low stock items\n"
                   "Please choose (1-5):",
        'options': [None, 'top5', 'low_cost', 'most_profitable', 'low_stock']
    }

def generate_chat_graph(metric, chart_type=None, filter_option=None):
    """Generate and return HTML for the requested graph with enhanced options"""
    try:
        # Get advanced options for the metric
        advanced_options = get_advanced_graph_options(metric)
        
        # Set defaults if not provided
        if not chart_type:
            chart_type = advanced_options['default_chart']
        if not filter_option:
            filter_option = next(iter(advanced_options['filters'].values()))
        
        # Validate parameters
        validate_graph_parameters(metric, chart_type, filter_option)
        
        # Prepare and generate chart
        chart_config = prepare_chart_data(
            chart_type=chart_type,
            data_type=metric,
            filter_option=filter_option
        )
        
        # Generate statistics
        stats = {
            'count': len(chart_config['values']),
            'max': max(chart_config['values']) if chart_config['values'] else 0,
            'min': min(chart_config['values']) if chart_config['values'] else 0,
            'average': sum(chart_config['values'])/len(chart_config['values']) if chart_config['values'] else 0,
            'total': sum(chart_config['values']) if chart_config['values'] else 0
        }
        
        graph_html = generate_plotly_chart(chart_config)
        
        return {
            'html': graph_html,
            'stats': stats,
            'title': f"{metric.capitalize()} Analysis",
            'chart_type': chart_type,
            'filter': filter_option,
            'unit': chart_config.get('unit', '')  # Include the unit from chart config
        }
        
    except Exception as e:
        print(f"Error generating graph: {e}")
        raise

def get_advanced_graph_options(metric):
    """Return advanced options based on selected metric"""
    options = {
        'sales': {
            'filters': {
                'Top Selling': 'top5',
                'Low Stock': 'low_stock',
                'All Items': None
            },
            'default_chart': 'bar'
        },
        'price': {
            'filters': {
                'Most Expensive': 'top5',
                'Most Affordable': 'affordable',
                'All Prices': None
            },
            'default_chart': 'bar'
        },
        'cost': {
            'filters': {
                'Highest Cost': 'top5',
                'Lowest Cost': 'low_cost',
                'All Costs': None
            },
            'default_chart': 'bar'
        },
        'profit': {
            'filters': {
                'Most Profitable': 'most_profitable',
                'Least Profitable': 'low_profit',
                'All Profits': None
            },
            'default_chart': 'bar'
        }
    }
    
    return options.get(metric, {
        'filters': {'All Items': None},
        'default_chart': 'bar'
    })

def validate_graph_parameters(metric, chart_type, filter_option):
    """Validate the graph parameters before generation"""
    valid_metrics = ['sales', 'price', 'cost', 'profit']
    valid_chart_types = ['bar', 'line', 'pie', 'doughnut']
    valid_filters = [None, 'top5', 'low_stock', 'affordable', 'low_cost', 'most_profitable', 'low_profit']
    
    if metric not in valid_metrics:
        raise ValueError("Invalid metric selected")
    if chart_type not in valid_chart_types:
        raise ValueError("Invalid chart type selected")
    if filter_option not in valid_filters:
        raise ValueError("Invalid filter option selected")
        
    return True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/logs")
@admin_required
@with_db_connection
def show_logs(conn):
    """
    Display query logs - now protected with admin access control.
    Only users with 'admin' role can access this page.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, user_query, bot_response FROM query_logs
            ORDER BY timestamp DESC
        """)
        logs = cursor.fetchall()
        return render_template("logs.html", logs=logs)
    except Error as e:
        print(f"Error fetching logs: {e}")
        return render_template("logs.html", logs=[])

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "").strip()
        current_state = session.get('conversation_state', STATE_NORMAL)
        temp_item = session.get('temp_item', {})

        # Session validation at the start
        if current_state != STATE_NORMAL:
            if (current_state == STATE_SELECT_METRIC and 'graph_options' not in session) or \
               (current_state == STATE_SELECT_CHART_TYPE and 'chart_options' not in session) or \
               (current_state == STATE_SELECT_FILTER and 'filter_options' not in session):
                session.clear()
                return jsonify({
                    "response": "Session expired. Please start over."
                })

        # Handle cancellation
        if user_message.lower() == 'cancel':
            session.clear()
            return jsonify({
                "response": "Operation cancelled"
            })

        # Complete normal conversation state implementation
        if current_state == STATE_NORMAL:
            # Handle graph requests
            if detect_graph_request(user_message):
                session['conversation_state'] = STATE_SELECT_METRIC
                options = get_graph_metric_options()
                session['graph_options'] = options['options']
                return jsonify({
                    "response": options['response']
                })
            
            # Handle regular queries (price, sales, cost, etc.)
            query_type, item_name = detect_query_type(user_message)
            
            if query_type in ['price', 'sales', 'cost']:
                db_item_name, value, sql_used, is_typo = get_item_info(item_name, f"item_{query_type}")
                response = format_response(query_type, item_name, db_item_name, value, is_typo)
                
                # Log the query
                if sql_used:
                    log_query(user_message, sql_used, response)
                
                return jsonify({
                    "response": response
                })
            
            # Handle add item requests
            new_item_name = detect_add_item_request(user_message)
            if new_item_name:
                session['conversation_state'] = STATE_ADD_ITEM_SALES
                session['temp_item'] = {'name': new_item_name}
                return jsonify({
                    "response": f"Great! I'll help you add '{new_item_name}'. How many units have been sold?"
                })
            
            # Default response for unrecognized queries
            return jsonify({
                "response": "I can help you with:\n- Item prices: 'What's the price of pen?'\n- Sales data: 'How many notebooks sold?'\n- Production costs: 'Cost of ruler?'\n- Adding items: 'Add new item laptop'\n- Creating graphs: 'Show me a graph'"
            })

        # Handle add item flow states
        elif current_state == STATE_ADD_ITEM_SALES:
            try:
                sales = int(user_message)
                temp_item['sales'] = sales
                session['temp_item'] = temp_item
                session['conversation_state'] = STATE_ADD_ITEM_PRICE
                return jsonify({
                    "response": f"Sales recorded: {sales} units. What's the selling price?"
                })
            except ValueError:
                return jsonify({
                    "response": "Please enter a valid number for sales quantity."
                })

        elif current_state == STATE_ADD_ITEM_PRICE:
            try:
                price = float(user_message)
                temp_item['price'] = price
                session['temp_item'] = temp_item
                session['conversation_state'] = STATE_ADD_ITEM_COST
                return jsonify({
                    "response": f"Price recorded: ${price:.2f}. What's the production cost?"
                })
            except ValueError:
                return jsonify({
                    "response": "Please enter a valid price (e.g., 5.99)."
                })

        elif current_state == STATE_ADD_ITEM_COST:
            try:
                cost = float(user_message)
                temp_item['cost'] = cost
                
                # Add the item to database
                success, message = add_new_item(None, temp_item)
                
                # Clear session
                session.clear()
                
                return jsonify({
                    "response": message
                })
            except ValueError:
                return jsonify({
                    "response": "Please enter a valid cost (e.g., 2.50)."
                })

        # Graph generation states
        elif current_state == STATE_SELECT_METRIC:
            try:
                choice = int(user_message)
                if 1 <= choice <= 4:
                    selected_metric = session['graph_options'][choice-1]
                    session['selected_metric'] = selected_metric
                    session['conversation_state'] = STATE_SELECT_CHART_TYPE
                    options = get_chart_type_options()
                    session['chart_options'] = options['options']
                    return jsonify({
                        "response": options['response']
                    })
                raise ValueError
            except ValueError:
                return jsonify({
                    "response": "Please enter a number between 1-4"
                })

        elif current_state == STATE_SELECT_CHART_TYPE:
            try:
                choice = int(user_message)
                if 1 <= choice <= 4:
                    selected_chart = session['chart_options'][choice-1]
                    session['selected_chart'] = selected_chart
                    session['conversation_state'] = STATE_SELECT_FILTER
                    options = get_filter_options()
                    session['filter_options'] = options['options']
                    return jsonify({
                        "response": options['response']
                    })
                raise ValueError
            except ValueError:
                return jsonify({
                    "response": "Please enter a number between 1-4"
                })

        elif current_state == STATE_SELECT_FILTER:
            try:
                choice = int(user_message)
                if 1 <= choice <= 5:
                    selected_filter = session['filter_options'][choice-1]
                    selected_metric = session.get('selected_metric')
                    selected_chart = session.get('selected_chart')
                    
                    try:
                        graph_data = generate_chat_graph(
                            metric=selected_metric,
                            chart_type=selected_chart,
                            filter_option=selected_filter
                        )
                        
                        # Format stats for display
                        stats_text = (
                            f"\n📊 Graph Statistics:\n"
                            f"• Items shown: {graph_data['stats']['count']}\n"
                            f"• Highest value: {graph_data['unit']}{graph_data['stats']['max']:.2f}\n"
                            f"• Lowest value: {graph_data['unit']}{graph_data['stats']['min']:.2f}\n"
                            f"• Average: {graph_data['unit']}{graph_data['stats']['average']:.2f}\n"
                            f"• Total: {graph_data['unit']}{graph_data['stats']['total']:.2f}"
                        )
                        
                        session.clear()
                        return jsonify({
                            "response": f"Here's your {selected_metric} graph:{stats_text}",
                            "graph": graph_data['html']
                        })
                        
                    except Exception as e:
                        session.clear()
                        return jsonify({
                            "response": f"Sorry, I couldn't generate that graph. Error: {str(e)}"
                        })
                        
                raise ValueError
            except ValueError:
                return jsonify({
                    "response": "Please enter a number between 1-5"
                })

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        session.clear()
        return jsonify({
            "response": "Sorry, something went wrong. Please try again."
        })

@app.route("/login", methods=["GET", "POST"])
@with_db_connection
def login(conn):
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        
        cursor = conn.cursor()
        cursor.execute("SELECT username, password_hash, role FROM users WHERE LOWER(username) = LOWER(?)", (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user[1], password):
            # FIXED: Store only username string, not entire user tuple
            session['username'] = user[0]  # Just the username string
            session['role'] = user[2]      # Just the role string
            flash("Login successful!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password", "error")
    
    return render_template("login.html", form=form)

@app.route("/register", methods=["GET", "POST"])
@with_db_connection
def register(conn):
    """
    Enhanced registration route that processes clearance level selection
    and grants appropriate role based on admin code verification.
    """
    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        selected_clearance = form.clearance_level.data
        
        cursor = conn.cursor()
        
        # Check for existing username (case-insensitive)
        cursor.execute("SELECT username FROM users")
        existing_usernames = [u[0].lower() for u in cursor.fetchall()]
        
        if username.lower() in existing_usernames:
            flash("Username already taken. Please choose another.", "error")
        else:
            # Hash the password
            hashed_password = generate_password_hash(password)
            
            # Insert user with the selected role
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, hashed_password, selected_clearance)
            )
            conn.commit()
            
            # Provide role-specific success messages
            if selected_clearance == 'admin':
                flash("Admin account created successfully! You now have full access to all features.", "success")
            else:
                flash("User account created successfully! Please log in to continue.", "success")
            
            return redirect(url_for('login'))
    
    return render_template("register.html", form=form)

@app.route("/change_password", methods=["GET", "POST"])
@with_db_connection
def change_password(conn):
    if 'username' not in session:
        flash("Please log in to change your password.", "warning")
        return redirect(url_for('login'))

    form = ChangePasswordForm()
    if form.validate_on_submit():
        username = session['username']
        current_password = form.current_password.data
        new_password = form.new_password.data

        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user[0], current_password):
            new_hash = generate_password_hash(new_password)
            cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username))
            conn.commit()
            flash("Password updated successfully.", "success")
            return redirect(url_for('index'))
        else:
            flash("Current password is incorrect.", "error")

    return render_template("change_password.html", form=form)

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

def generate_chart_data(conn, chart_type, data_type, filter_option=None, limit=None):
    """
    Generate chart data with advanced filtering and options.
    Returns chart data and configuration for Chart.js with additional features.
    """
    cursor = conn.cursor()
    try:
        # Handle different filtering and limiting options
        limit_clause = ""
        if limit and limit.isdigit():
            limit_clause = f"LIMIT {limit}"
            
        if data_type == "sales":
            # Get sales data with optional filtering
            if filter_option == "top5":
                cursor.execute("SELECT item_name, item_sales FROM items ORDER BY item_sales DESC LIMIT 5")
            elif filter_option == "low_stock":
                cursor.execute("SELECT item_name, item_sales FROM items WHERE item_sales < 100 ORDER BY item_sales ASC")
            else:
                cursor.execute(f"SELECT item_name, item_sales FROM items ORDER BY item_sales DESC {limit_clause}")
            
            data = cursor.fetchall()
            labels = [item[0] for item in data]
            values = [item[1] for item in data]
            title = "Sales by Product"
            unit = "units"
            
        elif data_type == "profit":
            # Calculate profit with filtering options
            if filter_option == "top5":
                cursor.execute("SELECT item_name, (item_price - item_cost) as profit FROM items ORDER BY profit DESC LIMIT 5")
            elif filter_option == "most_profitable":
                cursor.execute("SELECT item_name, (item_price - item_cost) as profit FROM items WHERE (item_price - item_cost) > 2.00 ORDER BY profit DESC")
            else:
                cursor.execute(f"SELECT item_name, (item_price - item_cost) as profit FROM items ORDER BY profit DESC {limit_clause}")
            
            data = cursor.fetchall()
            labels = [item[0] for item in data]
            values = [round(item[1], 2) for item in data]
            title = "Profit by Product"
            unit = "$"
            
        elif data_type == "price":
            # Get price data with filtering
            if filter_option == "top5":
                cursor.execute("SELECT item_name, item_price FROM items ORDER BY item_price DESC LIMIT 5")
            elif filter_option == "affordable":
                cursor.execute("SELECT item_name, item_price FROM items WHERE item_price < 3.00 ORDER BY item_price ASC")
            else:
                cursor.execute(f"SELECT item_name, item_price FROM items ORDER BY item_price DESC {limit_clause}")
            
            data = cursor.fetchall()
            labels = [item[0] for item in data]
            values = [item[1] for item in data]
            title = "Price by Product"
            unit = "$"
            
        elif data_type == "cost":
            # Get cost data with filtering
            if filter_option == "top5":
                cursor.execute("SELECT item_name, item_cost FROM items ORDER BY item_cost DESC LIMIT 5")
            elif filter_option == "low_cost":
                cursor.execute("SELECT item_name, item_cost FROM items WHERE item_cost < 1.00 ORDER BY item_cost ASC")
            else:
                cursor.execute(f"SELECT item_name, item_cost FROM items ORDER BY item_cost DESC {limit_clause}")
            
            data = cursor.fetchall()
            labels = [item[0] for item in data]
            values = [item[1] for item in data]
            title = "Cost by Product"
            unit = "$"
            
        else:
            # Default to sales data
            cursor.execute(f"SELECT item_name, item_sales FROM items ORDER BY item_sales DESC {limit_clause}")
            data = cursor.fetchall()
            labels = [item[0] for item in data]
            values = [item[1] for item in data]
            title = "Sales by Product"
            unit = "units"

        # Calculate additional statistics
        if values:  # Only if we have data
            total_value = sum(values)
            average_value = total_value / len(values)
            max_value = max(values)
            min_value = min(values)
        else:
            total_value = average_value = max_value = min_value = 0

        # Prepare enhanced chart configuration
        chart_config = {
            'type': chart_type,
            'title': title,
            'labels': labels,
            'values': values,
            'unit': unit,
            'stats': {
                'total': total_value,
                'average': average_value,
                'max': max_value,
                'min': min_value,
                'count': len(values)
            }
        }

        return data, chart_config

    except Error as e:
        print(f"Error generating chart data: {e}")
        return None, None

def get_dashboard_stats(cursor):
    cursor.execute("SELECT COUNT(*) FROM items")
    total_items = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(item_sales) FROM items")
    total_sales = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(item_price - item_cost) FROM items")
    avg_profit = cursor.fetchone()[0] or 0
    
    return {
        'total_items': total_items,
        'total_sales': total_sales,
        'avg_profit': avg_profit
    }

@app.route('/graphs', methods=['GET', 'POST'])
def graphs():
    chart_html = None
    chart_config = None
    dashboard_stats = None
    
    if request.method == 'POST':
        chart_type = request.form.get('chart_type')
        data_type = request.form.get('data_type')
        filter_option = request.form.get('filter_option')
        limit = request.form.get('limit')
        
        # Convert limit to int if provided
        limit = int(limit) if limit and limit.isdigit() else None
        
        # Prepare chart data
        chart_config = prepare_chart_data(chart_type, data_type, filter_option, limit)
        chart_html = generate_plotly_chart(chart_config)
    
    # FIXED: Dashboard stats calculation with proper tuple handling
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Fix: Properly handle the fetchone() results by extracting [0] element
        cursor.execute("SELECT COUNT(*) FROM items")
        result = cursor.fetchone()
        total_items = result[0] if result else 0
        
        cursor.execute("SELECT SUM(item_sales) FROM items")
        result = cursor.fetchone()
        total_sales = result[0] if result and result[0] is not None else 0
        
        cursor.execute("SELECT AVG(item_price - item_cost) FROM items")
        result = cursor.fetchone()
        avg_profit = result[0] if result and result is not None else 0.0
        
        # Ensure all values are proper numbers, not tuples
        dashboard_stats = {
            'total_items': int(total_items),
            'total_sales': int(total_sales), 
            'avg_profit': float(avg_profit)
        }
        
    except Exception as e:
        print("Error fetching dashboard stats:", e)
        # Provide default values in case of error
        dashboard_stats = {
            'total_items': 0,
            'total_sales': 0,
            'avg_profit': 0.0
        }
    
    return render_template(
        'graphs.html',
        chart_html=chart_html,
        chart_config=chart_config,
        dashboard_stats=dashboard_stats
    )

@app.route("/api/chart", methods=["POST"])
def api_chart():
    payload   = request.json or {}
    data_type = payload.get("data_type")
    chart_type= payload.get("chart_type")
    filter_opt= payload.get("filter_option")
    limit     = payload.get("limit")

    chart_config = prepare_chart_data(chart_type, data_type, filter_opt, limit)
    chart_html   = generate_plotly_chart(chart_config)
    return jsonify({ "chart_html": chart_html })


if __name__ == "__main__":
    initialize_database()
    try:
        app.run(debug=True)
    finally:
        Database().close()
