# youtube_capstone_project

The **youtube_capstone_project** is designed for harvesting data from YouTube, providing a structured approach for data collection, transformation, and analysis. The project is divided into three distinct sections within a Streamlit application to simplify both user and developer interaction.

## Project Sections

1. **Youtube_Data_Harvesting**:
    - This section focuses on pulling data from YouTube by providing channel IDs as input and pushing it into a MongoDB Atlas database.
    - Users can input the number of YouTube channels they want to assess.
    - Input boxes are dynamically created for channel IDs based on the number input.
    - After entering the channel IDs, users can click "Get Details" to proceed.
    - Channel details, playlists, playlist items, and video details are extracted and saved in MongoDB.
    - A completion message is displayed along with navigation guidance to the next section in the sidebar.

2. **Creating DataFrames and Loading the SQL Database**:
    - In this section, data stored in MongoDB is extracted and structured into DataFrames for further analysis.
    - Channel details are saved as-is in a DataFrame.
    - Playlist details and playlist item details are merged based on the playlist ID to create a consolidated DataFrame with all necessary fields.
    - Video details are transformed, including converting the published date column to datetime.
    - A DataFrame for comment details is created from the MongoDB data.
    - Users are prompted to click a button to send all DataFrames to an SQL database.
    - The SQL database connection is established, and the data from DataFrames is loaded into the database.
    - DataFrames are displayed for reference, and users are guided to the next section.

3. **Querying SQL Database**:
    - In this section, various queries mentioned in the project description are executed one by one.
    - The query results are received as SQL tables and displayed as DataFrames.

For more information on using the YouTube API in Python, refer to [this guide](https://www.thepythoncode.com/article/using-youtube-api-in-python).

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/svramprabu/youtube_capstone_project
   cd youtube_capstone_project
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:

   ```bash
   streamlit run YouTube_Data_Harvesting.py
   ```

4. Follow the guided sections within the application to harvest, process, and analyze YouTube data.

## Contributors

- Ramprabu S V
<!-- - Another Contributor (if applicable) -->

## License

This project is a free to use/clone public repository.

---

Feel free to contribute to this project or report any issues on the [GitHub repository](https://github.com/svramprabu/youtube_capstone_project). Your feedback and contributions are highly appreciated and will help enhance the functionality and usability of this project.