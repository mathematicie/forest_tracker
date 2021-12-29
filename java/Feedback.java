//Package name ---

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;

import com.google.android.material.textfield.TextInputLayout;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.UUID;

public class Feedback extends AppCompatActivity {

    TextInputLayout feedbackText;
    Button submit;
    RadioGroup radioGroup;
    RadioButton feedback, feature;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_feedback);


        if (getSupportActionBar()!=null)
            getSupportActionBar().hide();

        feedbackText = findViewById(R.id.feedbackText);
        submit = findViewById(R.id.submitFeedback);
        radioGroup = findViewById(R.id.radioGroup1);

        feedback = findViewById(R.id.checkFeedback);
        feature = findViewById(R.id.checkFeature);

        feedback.setSelected(true);




        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                String textFeedback = feedbackText.getEditText().getText().toString();

                String ch = "";

                int id = radioGroup.getCheckedRadioButtonId();

                if (id==R.id.checkFeature){
                    ch = "Feature";
                } else if  (id==R.id.checkFeedback){
                    ch = "Feedback";
                }

                if (textFeedback.length()<10){
                    Toast.makeText(Feedback.this, "Enter at least 10 characters in feedback/feature", Toast.LENGTH_SHORT).show();
                } else {
                    String key = UUID.randomUUID().toString();

                    DatabaseReference reference;
                    reference = FirebaseDatabase.getInstance().getReference();
                    reference.child(ch).child(key).setValue(textFeedback);

                    Intent intent = new Intent(Feedback.this, MainActivity.class);
                    startActivity(intent);
                    finish();
                }
            }
        });



    }
}
