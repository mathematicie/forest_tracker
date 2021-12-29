//Package name ---

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.SharedPreferences;
import android.media.Image;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.ArrayList;

public class SeeAllDataByUser extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_see_all_data_by_user);

        TextView emptyText;
        ImageView image;

        emptyText = findViewById(R.id.showEmptyTextOnAllData);
        image = findViewById(R.id.showImageOnAllData);

        RecyclerView recyclerView = findViewById(R.id.seeDataByUserRecyclerView);

        recyclerView.setVisibility(View.GONE);
        emptyText.setVisibility(View.GONE);
        image.setVisibility(View.GONE);



        if (getSupportActionBar()!=null)
            getSupportActionBar().hide();

        ArrayList<SaveTreeClass> list;

        SharedPreferences sharedPreferences = getSharedPreferences("details", MODE_PRIVATE);
        Gson gson = new Gson();
        String json = sharedPreferences.getString("allUserData", null);
        Type type = new TypeToken<ArrayList<SaveTreeClass>>(){}.getType();

        list = gson.fromJson(json, type);

        if(list != null && list.size()>0) {
            recyclerView.setVisibility(View.VISIBLE);
            emptyText.setVisibility(View.GONE);
            image.setVisibility(View.GONE);
            ShowAllUserDataAdapter adapter = new ShowAllUserDataAdapter(list);
            recyclerView.setLayoutManager(new LinearLayoutManager(this));
            recyclerView.setAdapter(adapter);
        } else {
            recyclerView.setVisibility(View.GONE);
            emptyText.setVisibility(View.VISIBLE);
            image.setVisibility(View.VISIBLE);
        }

    }
}
