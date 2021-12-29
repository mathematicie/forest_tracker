package com.environment.foresttracker;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.OnProgressListener;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Type;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;

public class UploadImageProof extends AppCompatActivity {

    Uri filePath;
    ImageView imageView;
    StorageReference storageRef;


    TextView showLocation,showDistrict,  showData, showImageSet;
    Button selectImage, uploadImage;

    String StateUT, District, cutorplant;
    int value, userTreeValue;

    FirebaseDatabase database;
    DatabaseReference treeRef;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_upload_image_proof);

        database = FirebaseDatabase.getInstance();
        treeRef = database.getReference().child("Trees")
                .child("Locations");

        getSupportActionBar().hide();


        value = getIntent().getIntExtra("value", 0);
        StateUT = getIntent().getStringExtra("StateUT");
        District = getIntent().getStringExtra("District");
        cutorplant = getIntent().getStringExtra("cutorplant");
        userTreeValue = getIntent().getIntExtra("userTreeValue", 0);


        selectImage = findViewById(R.id.selectImageButton);
        uploadImage = findViewById(R.id.uploadImageButton);



        imageView = findViewById(R.id.imageView);


        storageRef = FirebaseStorage.getInstance().getReference();

        showLocation = findViewById(R.id.showLocationOnImageProofActivity);
        showData = findViewById(R.id.showDataToUploadImageActivity);

        showLocation.setText(StateUT);


        showDistrict = findViewById(R.id.showDistrictOnImageProofActivity);

        showDistrict.setText(District);

        String toShow = "";
        if (cutorplant.equals("cut")) toShow = "Trees Cut: ";
        else if (cutorplant.equals("plant")) toShow = "Trees Planted: ";

        showData.setText("Data to upload:\n"+toShow+String.valueOf(userTreeValue));

        selectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickImage();
            }
        });

        uploadImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(filePath==null){
                    Toast.makeText(UploadImageProof.this, "Select an image first.", Toast.LENGTH_SHORT).show();
                }
                uploadImageFirebase();
            }
        });


    }

    public void pickImage(){

        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Image "), 100);

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode==100 && resultCode==RESULT_OK && data!=null && data.getData()!=null){

            filePath = data.getData();

            try{
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), filePath);
                imageView.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

    }

    Calendar calendar;
    String dateTime;
    SimpleDateFormat simpleDateFormat;

    public void uploadImageFirebase(){

        if (filePath!=null){

            ProgressDialog progressDialog = new ProgressDialog(this);
            progressDialog.setTitle("Uploading...");
            progressDialog.show();

            calendar = Calendar.getInstance();
            simpleDateFormat = new SimpleDateFormat("dd-MM-yyyy_HH-mm-ss_aaa");
            dateTime = simpleDateFormat.format(calendar.getTime()).toString();


            String toShow = "";
            if (cutorplant.equals("cut")) toShow = "Trees Cut";
            else if (cutorplant.equals("plant")) toShow = "Trees Planted";

            StorageReference storageReference = storageRef.child("images/"+ StateUT+"_"
                    + District + "_" + String.valueOf(userTreeValue) +" "+ toShow + "_" + dateTime);

            storageReference.putFile(filePath).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                    String treescutorplanted = "";
                    if (cutorplant.equals("cut")) treescutorplanted = "TreesCut";
                    else if (cutorplant.equals("plant")) treescutorplanted = "TreesPlanted";


                    treeRef.child(StateUT).child(District).child(treescutorplanted).setValue(value);

                    progressDialog.dismiss();
                    Toast.makeText(UploadImageProof.this, "Image Uploaded", Toast.LENGTH_SHORT).show();

                    SharedPreferences sharedPreferences = getSharedPreferences("details", MODE_PRIVATE);
                    SharedPreferences.Editor editor = sharedPreferences.edit();

                    Gson getGson = new Gson();
                    String getJson = sharedPreferences.getString("allUserData", null);
                    Type type = new TypeToken<ArrayList<SaveTreeClass>>(){}.getType();


                    ArrayList<SaveTreeClass> list;
                    list = new ArrayList<>();

                    list = getGson.fromJson(getJson, type);


                    if (list != null){
                        list.add(new SaveTreeClass(StateUT, District, String.valueOf(userTreeValue), cutorplant, dateTime.replace("_", " ")));
                    }
                    else {
                        list = new ArrayList<>();
                        list.add(new SaveTreeClass(StateUT, District, String.valueOf(userTreeValue), cutorplant, dateTime.replace("_", " ")));
                    }


                    Gson gson = new Gson();

                    String json = gson.toJson(list);
                    editor.putString("allUserData", json);
                    editor.commit();

                    Intent intent = new Intent(UploadImageProof.this, MainActivity.class);
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                    startActivity(intent);
                    finish();
                }
            }).addOnProgressListener(new OnProgressListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onProgress(@NonNull UploadTask.TaskSnapshot snapshot) {
                    double progress = (100.0*snapshot.getBytesTransferred()/snapshot.getTotalByteCount());
                    progressDialog.setMessage(String.valueOf((int)progress+" % uploaded"));
                }
            });

        }

    }
}