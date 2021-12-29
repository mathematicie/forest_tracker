//Package name ---

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class ValidateDataAndSubmit extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_validate_data_and_submit);

        getSupportActionBar().hide();

        int value = getIntent().getIntExtra("value", 0);
        String StateUT = getIntent().getStringExtra("StateUT");
        String District = getIntent().getStringExtra("District");
        String cutorplant = getIntent().getStringExtra("cutorplant");
        int userTreeValue = getIntent().getIntExtra("userTreeValue", 0);

        Button solveMath, image, treeNumber;

        solveMath = findViewById(R.id.solveMathButton);
        image = findViewById(R.id.imageProofButton);
        treeNumber = findViewById(R.id.treeNumberButton);

        solveMath.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ValidateDataAndSubmit.this, SolveMath.class);
                intent.putExtra("value", value);
                intent.putExtra("StateUT", StateUT);
                intent.putExtra("District", District);
                intent.putExtra("cutorplant", cutorplant);
                intent.putExtra("userTreeValue", userTreeValue);

                startActivity(intent);
            }
        });

        image.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ValidateDataAndSubmit.this, UploadImageProof.class);
                intent.putExtra("value", value);
                intent.putExtra("StateUT", StateUT);
                intent.putExtra("District", District);
                intent.putExtra("cutorplant", cutorplant);
                intent.putExtra("userTreeValue", userTreeValue);


                startActivity(intent);
            }
        });

        treeNumber.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ValidateDataAndSubmit.this, EnterTreeNumber.class);
                intent.putExtra("value", value);
                intent.putExtra("StateUT", StateUT);
                intent.putExtra("District", District);
                intent.putExtra("cutorplant", cutorplant);
                intent.putExtra("userTreeValue", userTreeValue);


                startActivity(intent);
            }
        });


    }
}
