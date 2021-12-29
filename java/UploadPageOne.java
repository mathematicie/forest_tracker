//Package name ---

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;

import java.util.ArrayList;

public class UploadPageOne extends AppCompatActivity {

    ArrayList<String> list = new ArrayList<>();

    Spinner spinner, spinner2;

    ArrayAdapter adapter, adapter2;


    ArrayList<String> AndamanandNicobarIsland, AndhraPradesh, ArunachalPradesh, Assam, Bihar, Chandigarh, Chhattisgarh,
            DadraandNagarHaveli, DamanandDiu, Delhi, Goa, Gujarat, Haryana, HimachalPradesh, JammuandKashmir, Jharkhand,
            Karnataka, Kerala, Ladakh, Lakshadweep, MadhyaPradesh, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland,
            Odisha, Puducherry, Punjab, Rajasthan, Sikkim, TamilNadu, Telangana, Tripura, Uttarakhand, UttarPradesh, WestBengal;


    String StateUT, locality;

    ArrayList<String> mainList;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_upload_page_one);

        getSupportActionBar().hide();

        mainList=new ArrayList<>();

        list.add("Andaman and Nicobar Islands");
        list.add("Andhra Pradesh");
        list.add("Arunachal Pradesh");
        list.add("Assam");
        list.add("Bihar");
        list.add("Chandigarh");
        list.add("Chhattisgarh");
        list.add("Dadra and Nagar Haveli");
        list.add("Daman and Diu");
        list.add("Delhi");
        list.add("Goa");
        list.add("Gujarat");
        list.add("Haryana");
        list.add("Himachal Pradesh");
        list.add("Jammu and Kashmir");
        list.add("Jharkhand");
        list.add("Karnataka");
        list.add("Kerala");
        list.add("Ladakh");
        list.add("Lakshadweep");
        list.add("Madhya Pradesh");
        list.add("Maharashtra");
        list.add("Manipur");
        list.add("Meghalaya");
        list.add("Mizoram");
        list.add("Nagaland");
        list.add("Odisha");
        list.add("Puducherry");
        list.add("Punjab");
        list.add("Rajasthan");
        list.add("Sikkim");
        list.add("Tamil Nadu");
        list.add("Telangana");
        list.add("Tripura");
        list.add("Uttar Pradesh");
        list.add("Uttarakhand");
        list.add("West Bengal");


        AndamanandNicobarIsland= new ArrayList<>();
        AndhraPradesh= new ArrayList<>();
        ArunachalPradesh= new ArrayList<>();
        Assam= new ArrayList<>();
        Bihar= new ArrayList<>();
        Chandigarh= new ArrayList<>();
        Chhattisgarh= new ArrayList<>();
        DadraandNagarHaveli= new ArrayList<>();
        DamanandDiu= new ArrayList<>();
        Delhi= new ArrayList<>();
        Goa= new ArrayList<>();
        Gujarat= new ArrayList<>();
        Haryana= new ArrayList<>();
        HimachalPradesh= new ArrayList<>();
        JammuandKashmir= new ArrayList<>();
        Jharkhand= new ArrayList<>();
        Karnataka= new ArrayList<>();
        Kerala= new ArrayList<>();
        Ladakh= new ArrayList<>();
        Lakshadweep= new ArrayList<>();
        MadhyaPradesh= new ArrayList<>();
        Maharashtra= new ArrayList<>();
        Manipur= new ArrayList<>();
        Meghalaya= new ArrayList<>();
        Mizoram= new ArrayList<>();
        Nagaland= new ArrayList<>();
        Odisha= new ArrayList<>();
        Puducherry= new ArrayList<>();
        Punjab= new ArrayList<>();
        Rajasthan= new ArrayList<>();
        Sikkim= new ArrayList<>();
        TamilNadu= new ArrayList<>();
        Telangana= new ArrayList<>();
        Tripura= new ArrayList<>();
        Uttarakhand= new ArrayList<>();
        UttarPradesh= new ArrayList<>();
        WestBengal= new ArrayList<>();





        spinner = findViewById(R.id.StateUTSelect);
        spinner2 = findViewById(R.id.localitySelect);


        {
            AndamanandNicobarIsland.add("Nicobar");
            AndamanandNicobarIsland.add("North and Middle Andaman");
            AndamanandNicobarIsland.add("South Andaman");

            AndhraPradesh.add("Anantapur");
            AndhraPradesh.add("Chittoor");
            AndhraPradesh.add("East Godavari");
            AndhraPradesh.add("Guntur");
            AndhraPradesh.add("Krishna");
            AndhraPradesh.add("Kurnool");
            AndhraPradesh.add("Prakasam");
            AndhraPradesh.add("Srikakulam");
            AndhraPradesh.add("Sri Potti Sriramulu Nellore");
            AndhraPradesh.add("Visakhapatnam");
            AndhraPradesh.add("Vizianagaram");
            AndhraPradesh.add("West Godavari");
            AndhraPradesh.add("YSR District, Kadapa (Cuddapah)");

            ArunachalPradesh.add("Anjaw");
            ArunachalPradesh.add("Changlang");
            ArunachalPradesh.add("Dibang Valley");
            ArunachalPradesh.add("East Kameng");
            ArunachalPradesh.add("East Siang");
            ArunachalPradesh.add("Kamle");
            ArunachalPradesh.add("Kra Daadi");
            ArunachalPradesh.add("Kurung Kumey");
            ArunachalPradesh.add("Lepa Rada");
            ArunachalPradesh.add("Lohit");
            ArunachalPradesh.add("Longding");
            ArunachalPradesh.add("Lower Dibang Valley");
            ArunachalPradesh.add("Lower Siang");
            ArunachalPradesh.add("Lower Subansiri");
            ArunachalPradesh.add("Namsai");
            ArunachalPradesh.add("Pakke Kessang");
            ArunachalPradesh.add("Papum Pare");
            ArunachalPradesh.add("Shi Yomi");
            ArunachalPradesh.add("Siang");
            ArunachalPradesh.add("Tawang");
            ArunachalPradesh.add("Tirap");
            ArunachalPradesh.add("Upper Siang");
            ArunachalPradesh.add("Upper Subansiri");
            ArunachalPradesh.add("West Kameng");
            ArunachalPradesh.add("West Siang");


            Assam.add("Baksa");
            Assam.add("Barpeta");
            Assam.add("Biswanath");
            Assam.add("Bongaigaon");
            Assam.add("Cachar");
            Assam.add("Charaideo");
            Assam.add("Chirang");
            Assam.add("Darrang");
            Assam.add("Dhemaji");
            Assam.add("Dhubri");
            Assam.add("Dibrugarh");
            Assam.add("Dima Hasao (North Cachar Hills)");
            Assam.add("Goalpara");
            Assam.add("Golaghat");
            Assam.add("Hailakandi");
            Assam.add("Hojai");
            Assam.add("Jorhat");
            Assam.add("Kamrup");
            Assam.add("Kamrup Metropolitan");
            Assam.add("Karbi Anglong");
            Assam.add("Karimganj");
            Assam.add("Kokrajhar");
            Assam.add("Lakhimpur");
            Assam.add("Majuli");
            Assam.add("Morigaon");
            Assam.add("Nagaon");
            Assam.add("Nalbari");
            Assam.add("Sivasagar");
            Assam.add("Sonitpur");
            Assam.add("South Salamara-Mankachar");
            Assam.add("Tinsukia");
            Assam.add("Udalguri");
            Assam.add("West Karbi Anglong");


            Bihar.add("Araria");
            Bihar.add("Arwal");
            Bihar.add("Aurangabad");
            Bihar.add("Banka");
            Bihar.add("Begusarai");
            Bihar.add("Bhagalpur");
            Bihar.add("Bhojpur");
            Bihar.add("Buxar");
            Bihar.add("Darbhanga");
            Bihar.add("East Champaran (Motihari)");
            Bihar.add("Gaya");
            Bihar.add("Gopalganj");
            Bihar.add("Jamui");
            Bihar.add("Jehanabad");
            Bihar.add("Kaimur (Bhabua)");
            Bihar.add("Katihar");
            Bihar.add("Khagaria");
            Bihar.add("Kishanganj");
            Bihar.add("Lakhisarai");
            Bihar.add("Madhepura");
            Bihar.add("Madhubani");
            Bihar.add("Munger (Monghyr)");
            Bihar.add("Muzaffarpur");
            Bihar.add("Nalanda");
            Bihar.add("Nawada");
            Bihar.add("Patna");
            Bihar.add("Purnia (Purnea)");
            Bihar.add("Rohtas");
            Bihar.add("Saharsa");
            Bihar.add("Samastipur");
            Bihar.add("Saran");
            Bihar.add("Sheikhpura");
            Bihar.add("Sheohar");
            Bihar.add("Sitamarhi");
            Bihar.add("Siwan");
            Bihar.add("Supaul");
            Bihar.add("Vaishali");
            Bihar.add("West Champaran");


            Chandigarh.add("Chandigarh (UT)");


            Chhattisgarh.add("Balod");
            Chhattisgarh.add("Baloda Bazar");
            Chhattisgarh.add("Balrampur");
            Chhattisgarh.add("Bastar");
            Chhattisgarh.add("Bemetara");
            Chhattisgarh.add("Bijapur");
            Chhattisgarh.add("Bilaspur");
            Chhattisgarh.add("Dantewada (South Bastar)");
            Chhattisgarh.add("Dhamtari");
            Chhattisgarh.add("Durg");
            Chhattisgarh.add("Gariyaband");
            Chhattisgarh.add("Janjgir-Champa");
            Chhattisgarh.add("Jashpur");
            Chhattisgarh.add("Kabirdham (Kawardha)");
            Chhattisgarh.add("Kanker (North Bastar)");
            Chhattisgarh.add("Kondagaon");
            Chhattisgarh.add("Korba");
            Chhattisgarh.add("Korea (Koriya)");
            Chhattisgarh.add("Mahasamund");
            Chhattisgarh.add("Mungeli");
            Chhattisgarh.add("Narayanpur");
            Chhattisgarh.add("Raigarh");
            Chhattisgarh.add("Raipur");
            Chhattisgarh.add("Rajnandgaon");
            Chhattisgarh.add("Sukma");
            Chhattisgarh.add("Surajpur");
            Chhattisgarh.add("Surguja");


            DadraandNagarHaveli.add("Dadra and Nagar Haveli (UT)");


            DamanandDiu.add("Daman");
            DamanandDiu.add("Diu");

            Delhi.add("Central Delhi");
            Delhi.add("East Delhi");
            Delhi.add("New Delhi");
            Delhi.add("North Delhi");
            Delhi.add("North East Delhi");
            Delhi.add("North West Delhi");
            Delhi.add("Shahdara");
            Delhi.add("South Delhi");
            Delhi.add("South East Delhi");
            Delhi.add("South West Delhi");
            Delhi.add("West Delhi");

            Goa.add("North Goa");
            Goa.add("South Goa");

            Gujarat.add("Ahmedabad");
            Gujarat.add("Amreli");
            Gujarat.add("Anand");
            Gujarat.add("Aravalli");
            Gujarat.add("Banaskantha (Palanpur)");
            Gujarat.add("Bharuch");
            Gujarat.add("Bhavnagar");
            Gujarat.add("Botad");
            Gujarat.add("Chhota Udepur");
            Gujarat.add("Dahod");
            Gujarat.add("Dangs (Ahwa)");
            Gujarat.add("Devbhoomi Dwarka");
            Gujarat.add("Gandhinagar");
            Gujarat.add("Gir Somnath");
            Gujarat.add("Jamnagar");
            Gujarat.add("Junagadh");
            Gujarat.add("Kachchh");
            Gujarat.add("Kheda (Nadiad)");
            Gujarat.add("Mahisagar");
            Gujarat.add("Mehsana");
            Gujarat.add("Morbi");
            Gujarat.add("Narmada (Rajpipla)");
            Gujarat.add("Navsari");
            Gujarat.add("Panchmahal (Godhra)");
            Gujarat.add("Patan");
            Gujarat.add("Porbandar");
            Gujarat.add("Rajkot");
            Gujarat.add("Sabarkantha (Himmatnagar)");
            Gujarat.add("Surat");
            Gujarat.add("Surendranagar");
            Gujarat.add("Tapi (Vyara)");
            Gujarat.add("Vadodara");
            Gujarat.add("Valsad");

            Haryana.add("Ambala");
            Haryana.add("Bhiwani");
            Haryana.add("Charkhi Dadri");
            Haryana.add("Faridabad");
            Haryana.add("Fatehabad");
            Haryana.add("Gurugram (Gurgaon)");
            Haryana.add("Hisar");
            Haryana.add("Jhajjar");
            Haryana.add("Jind");
            Haryana.add("Kaithal");
            Haryana.add("Karnal");
            Haryana.add("Kurukshetra");
            Haryana.add("Mahendragarh");
            Haryana.add("Nuh");
            Haryana.add("Palwal");
            Haryana.add("Panchkula");
            Haryana.add("Panipat");
            Haryana.add("Rewari");
            Haryana.add("Rohtak");
            Haryana.add("Sirsa");
            Haryana.add("Sonipat");
            Haryana.add("Yamunanagar");

            HimachalPradesh.add("Bilaspur");
            HimachalPradesh.add("Chamba");
            HimachalPradesh.add("Hamirpur");
            HimachalPradesh.add("Kangra");
            HimachalPradesh.add("Kinnaur");
            HimachalPradesh.add("Kullu");
            HimachalPradesh.add("Lahaul & Spiti");
            HimachalPradesh.add("Mandi");
            HimachalPradesh.add("Shimla");
            HimachalPradesh.add("Sirmaur (Sirmour)");
            HimachalPradesh.add("Solan");
            HimachalPradesh.add("Una");


            JammuandKashmir.add("Anantnag");
            JammuandKashmir.add("Bandipore");
            JammuandKashmir.add("Baramulla");
            JammuandKashmir.add("Budgam");
            JammuandKashmir.add("Doda");
            JammuandKashmir.add("Ganderbal");
            JammuandKashmir.add("Jammu");
            JammuandKashmir.add("Kathua");
            JammuandKashmir.add("Kishtwar");
            JammuandKashmir.add("Kulgam");
            JammuandKashmir.add("Kupwara");
            JammuandKashmir.add("Poonch");
            JammuandKashmir.add("Pulwama");
            JammuandKashmir.add("Rajouri");
            JammuandKashmir.add("Ramban");
            JammuandKashmir.add("Reasi");
            JammuandKashmir.add("Samba");
            JammuandKashmir.add("Shopian");
            JammuandKashmir.add("Srinagar");
            JammuandKashmir.add("Udhampur");

            Jharkhand.add("Bokaro");
            Jharkhand.add("Chatra");
            Jharkhand.add("Deoghar");
            Jharkhand.add("Dhanbad");
            Jharkhand.add("Dumka");
            Jharkhand.add("East Singhbhum");
            Jharkhand.add("Garhwa");
            Jharkhand.add("Giridih");
            Jharkhand.add("Godda");
            Jharkhand.add("Gumla");
            Jharkhand.add("Hazaribag");
            Jharkhand.add("Jamtara");
            Jharkhand.add("Khunti");
            Jharkhand.add("Koderma");
            Jharkhand.add("Latehar");
            Jharkhand.add("Lohardaga");
            Jharkhand.add("Pakur");
            Jharkhand.add("Palamu");
            Jharkhand.add("Ramgarh");
            Jharkhand.add("Ranchi");
            Jharkhand.add("Sahibganj");
            Jharkhand.add("Seraikela-Kharsawan");
            Jharkhand.add("Simdega");
            Jharkhand.add("West Singhbhum");

            Karnataka.add("Bagalkot");
            Karnataka.add("Ballari (Bellary)");
            Karnataka.add("Belagavi (Belgaum)");
            Karnataka.add("Bengaluru (Bangalore) Rural");
            Karnataka.add("Bengaluru (Bangalore) Urban");
            Karnataka.add("Bidar");
            Karnataka.add("Chamarajanagar");
            Karnataka.add("Chikballapur");
            Karnataka.add("Chikkamagaluru (Chikmagalur)");
            Karnataka.add("Chitradurga");
            Karnataka.add("Dakshina Kannada");
            Karnataka.add("Davangere");
            Karnataka.add("Dharwad");
            Karnataka.add("Gadag");
            Karnataka.add("Hassan");
            Karnataka.add("Haveri");
            Karnataka.add("Kalaburagi (Gulbarga)");
            Karnataka.add("Kodagu");
            Karnataka.add("Kolar");
            Karnataka.add("Koppal");
            Karnataka.add("Mandya");
            Karnataka.add("Mysuru (Mysore)");
            Karnataka.add("Raichur");
            Karnataka.add("Ramanagara");
            Karnataka.add("Shivamogga (Shimoga)");
            Karnataka.add("Tumakuru (Tumkur)");
            Karnataka.add("Udupi");
            Karnataka.add("Uttara Kannada (Karwar)");
            Karnataka.add("Vijayapura (Bijapur)");
            Karnataka.add("Yadgir");


            Kerala.add("Alappuzha");
            Kerala.add("Ernakulam");
            Kerala.add("Idukki");
            Kerala.add("Kannur");
            Kerala.add("Kasaragod");
            Kerala.add("Kollam");
            Kerala.add("Kottayam");
            Kerala.add("Kozhikode");
            Kerala.add("Malappuram");
            Kerala.add("Palakkad");
            Kerala.add("Pathanamthitta");
            Kerala.add("Thiruvananthapuram");
            Kerala.add("Thrissur");
            Kerala.add("Wayanad");

            Ladakh.add("Kargil ");
            Ladakh.add("Leh");

            Lakshadweep.add("Lakshadweep (UT)");

            MadhyaPradesh.add("Agar Malwa");
            MadhyaPradesh.add("Alirajpur");
            MadhyaPradesh.add("Anuppur");
            MadhyaPradesh.add("Ashoknagar");
            MadhyaPradesh.add("Balaghat");
            MadhyaPradesh.add("Barwani");
            MadhyaPradesh.add("Betul");
            MadhyaPradesh.add("Bhind");
            MadhyaPradesh.add("Bhopal");
            MadhyaPradesh.add("Burhanpur");
            MadhyaPradesh.add("Chhatarpur");
            MadhyaPradesh.add("Chhindwara");
            MadhyaPradesh.add("Damoh");
            MadhyaPradesh.add("Datia");
            MadhyaPradesh.add("Dewas");
            MadhyaPradesh.add("Dhar");
            MadhyaPradesh.add("Dindori");
            MadhyaPradesh.add("Guna");
            MadhyaPradesh.add("Gwalior");
            MadhyaPradesh.add("Harda");
            MadhyaPradesh.add("Hoshangabad");
            MadhyaPradesh.add("Indore");
            MadhyaPradesh.add("Jabalpur");
            MadhyaPradesh.add("Jhabua");
            MadhyaPradesh.add("Katni");
            MadhyaPradesh.add("Khandwa");
            MadhyaPradesh.add("Khargone");
            MadhyaPradesh.add("Mandla");
            MadhyaPradesh.add("Mandsaur");
            MadhyaPradesh.add("Morena");
            MadhyaPradesh.add("Narsinghpur");
            MadhyaPradesh.add("Neemuch");
            MadhyaPradesh.add("Panna");
            MadhyaPradesh.add("Raisen");
            MadhyaPradesh.add("Rajgarh");
            MadhyaPradesh.add("Ratlam");
            MadhyaPradesh.add("Rewa");
            MadhyaPradesh.add("Sagar");
            MadhyaPradesh.add("Satna");
            MadhyaPradesh.add("Sehore");
            MadhyaPradesh.add("Seoni");
            MadhyaPradesh.add("Shahdol");
            MadhyaPradesh.add("Shajapur");
            MadhyaPradesh.add("Sheopur");
            MadhyaPradesh.add("Shivpuri");
            MadhyaPradesh.add("Sidhi");
            MadhyaPradesh.add("Singrauli");
            MadhyaPradesh.add("Tikamgarh");
            MadhyaPradesh.add("Ujjain");
            MadhyaPradesh.add("Umaria");
            MadhyaPradesh.add("Vidisha");


            Maharashtra.add("Ahmednagar");
            Maharashtra.add("Akola");
            Maharashtra.add("Amravati");
            Maharashtra.add("Aurangabad");
            Maharashtra.add("Beed");
            Maharashtra.add("Bhandara");
            Maharashtra.add("Buldhana");
            Maharashtra.add("Chandrapur");
            Maharashtra.add("Dhule");
            Maharashtra.add("Gadchiroli");
            Maharashtra.add("Gondia");
            Maharashtra.add("Hingoli");
            Maharashtra.add("Jalgaon");
            Maharashtra.add("Jalna");
            Maharashtra.add("Kolhapur");
            Maharashtra.add("Latur");
            Maharashtra.add("Mumbai City");
            Maharashtra.add("Mumbai Suburban");
            Maharashtra.add("Nagpur");
            Maharashtra.add("Nanded");
            Maharashtra.add("Nandurbar");
            Maharashtra.add("Nashik");
            Maharashtra.add("Osmanabad");
            Maharashtra.add("Palghar");
            Maharashtra.add("Parbhani");
            Maharashtra.add("Pune");
            Maharashtra.add("Raigad");
            Maharashtra.add("Ratnagiri");
            Maharashtra.add("Sangli");
            Maharashtra.add("Satara");
            Maharashtra.add("Sindhudurg");
            Maharashtra.add("Solapur");
            Maharashtra.add("Thane");
            Maharashtra.add("Wardha");
            Maharashtra.add("Washim");
            Maharashtra.add("Yavatmal");

            Manipur.add("Bishnupur");
            Manipur.add("Chandel");
            Manipur.add("Churachandpur");
            Manipur.add("Imphal East");
            Manipur.add("Imphal West");
            Manipur.add("Jiribam");
            Manipur.add("Kakching");
            Manipur.add("Kamjong");
            Manipur.add("Kangpokpi");
            Manipur.add("Noney");
            Manipur.add("Pherzawl");
            Manipur.add("Senapati");
            Manipur.add("Tamenglong");
            Manipur.add("Tengnoupal");
            Manipur.add("Thoubal");
            Manipur.add("Ukhrul");

            Meghalaya.add("East Garo Hills");
            Meghalaya.add("East Jaintia Hills");
            Meghalaya.add("East Khasi Hills");
            Meghalaya.add("North Garo Hills");
            Meghalaya.add("Ri Bhoi");
            Meghalaya.add("South Garo Hills");
            Meghalaya.add("South West Garo Hills");
            Meghalaya.add("South West Khasi Hills");
            Meghalaya.add("West Garo Hills");
            Meghalaya.add("West Jaintia Hills");
            Meghalaya.add("West Khasi Hills");

            Mizoram.add("Aizawl");
            Mizoram.add("Champhai");
            Mizoram.add("Kolasib");
            Mizoram.add("Lawngtlai");
            Mizoram.add("Lunglei");
            Mizoram.add("Mamit");
            Mizoram.add("Saiha");
            Mizoram.add("Serchhip");

            Nagaland.add("Dimapur");
            Nagaland.add("Kiphire");
            Nagaland.add("Kohima");
            Nagaland.add("Longleng");
            Nagaland.add("Mokokchung");
            Nagaland.add("Mon");
            Nagaland.add("Peren");
            Nagaland.add("Phek");
            Nagaland.add("Tuensang");
            Nagaland.add("Wokha");
            Nagaland.add("Zunheboto");

            Odisha.add("Angul");
            Odisha.add("Balangir");
            Odisha.add("Balasore");
            Odisha.add("Bargarh");
            Odisha.add("Bhadrak");
            Odisha.add("Boudh");
            Odisha.add("Cuttack");
            Odisha.add("Deogarh");
            Odisha.add("Dhenkanal");
            Odisha.add("Gajapati");
            Odisha.add("Ganjam");
            Odisha.add("Jagatsinghapur");
            Odisha.add("Jajpur");
            Odisha.add("Jharsuguda");
            Odisha.add("Kalahandi");
            Odisha.add("Kandhamal");
            Odisha.add("Kendrapara");
            Odisha.add("Kendujhar (Keonjhar)");
            Odisha.add("Khordha");
            Odisha.add("Koraput");
            Odisha.add("Malkangiri");
            Odisha.add("Mayurbhanj");
            Odisha.add("Nabarangpur");
            Odisha.add("Nayagarh");
            Odisha.add("Nuapada");
            Odisha.add("Puri");
            Odisha.add("Rayagada");
            Odisha.add("Sambalpur");
            Odisha.add("Sonepur");
            Odisha.add("Sundargarh");

            Puducherry.add("Karaikal");
            Puducherry.add("Mahe");
            Puducherry.add("Puducherry");
            Puducherry.add("Yanam");

            Punjab.add("Amritsar");
            Punjab.add("Barnala");
            Punjab.add("Bathinda");
            Punjab.add("Faridkot");
            Punjab.add("Fatehgarh Sahib");
            Punjab.add("Fazilka");
            Punjab.add("Ferozepur");
            Punjab.add("Gurdaspur");
            Punjab.add("Hoshiarpur");
            Punjab.add("Jalandhar");
            Punjab.add("Kapurthala");
            Punjab.add("Ludhiana");
            Punjab.add("Mansa");
            Punjab.add("Moga");
            Punjab.add("Muktsar");
            Punjab.add("Nawanshahr (Shahid Bhagat Singh Nagar)");
            Punjab.add("Pathankot");
            Punjab.add("Patiala");
            Punjab.add("Rupnagar");
            Punjab.add("Sahibzada Ajit Singh Nagar (Mohali)");
            Punjab.add("Sangrur");
            Punjab.add("Tarn Taran");

            Rajasthan.add("Ajmer");
            Rajasthan.add("Alwar");
            Rajasthan.add("Banswara");
            Rajasthan.add("Baran");
            Rajasthan.add("Barmer");
            Rajasthan.add("Bharatpur");
            Rajasthan.add("Bhilwara");
            Rajasthan.add("Bikaner");
            Rajasthan.add("Bundi");
            Rajasthan.add("Chittorgarh");
            Rajasthan.add("Churu");
            Rajasthan.add("Dausa");
            Rajasthan.add("Dholpur");
            Rajasthan.add("Dungarpur");
            Rajasthan.add("Hanumangarh");
            Rajasthan.add("Jaipur");
            Rajasthan.add("Jaisalmer");
            Rajasthan.add("Jalore");
            Rajasthan.add("Jhalawar");
            Rajasthan.add("Jhunjhunu");
            Rajasthan.add("Jodhpur");
            Rajasthan.add("Karauli");
            Rajasthan.add("Kota");
            Rajasthan.add("Nagaur");
            Rajasthan.add("Pali");
            Rajasthan.add("Pratapgarh");
            Rajasthan.add("Rajsamand");
            Rajasthan.add("Sawai Madhopur");
            Rajasthan.add("Sikar");
            Rajasthan.add("Sirohi");
            Rajasthan.add("Sri Ganganagar");
            Rajasthan.add("Tonk");
            Rajasthan.add("Udaipur");

            Sikkim.add("East Sikkim");
            Sikkim.add("North Sikkim");
            Sikkim.add("South Sikkim");
            Sikkim.add("West Sikkim");

            TamilNadu.add("Ariyalur");
            TamilNadu.add("Chengalpattu");
            TamilNadu.add("Chennai");
            TamilNadu.add("Coimbatore");
            TamilNadu.add("Cuddalore");
            TamilNadu.add("Dharmapuri");
            TamilNadu.add("Dindigul");
            TamilNadu.add("Erode");
            TamilNadu.add("Kallakurichi");
            TamilNadu.add("Kanchipuram");
            TamilNadu.add("Kanyakumari");
            TamilNadu.add("Karur");
            TamilNadu.add("Krishnagiri");
            TamilNadu.add("Madurai");
            TamilNadu.add("Nagapattinam");
            TamilNadu.add("Namakkal");
            TamilNadu.add("Nilgiris");
            TamilNadu.add("Perambalur");
            TamilNadu.add("Pudukkottai");
            TamilNadu.add("Ramanathapuram");
            TamilNadu.add("Ranipet");
            TamilNadu.add("Salem");
            TamilNadu.add("Sivaganga");
            TamilNadu.add("Tenkasi");
            TamilNadu.add("Thanjavur");
            TamilNadu.add("Theni");
            TamilNadu.add("Thoothukudi (Tuticorin)");
            TamilNadu.add("Tiruchirappalli");
            TamilNadu.add("Tirunelveli");
            TamilNadu.add("Tirupathur");
            TamilNadu.add("Tiruppur");
            TamilNadu.add("Tiruvallur");
            TamilNadu.add("Tiruvannamalai");
            TamilNadu.add("Tiruvarur");
            TamilNadu.add("Vellore");
            TamilNadu.add("Viluppuram");
            TamilNadu.add("Virudhunagar");


            Telangana.add("Adilabad");
            Telangana.add("Bhadradri Kothagudem");
            Telangana.add("Hyderabad");
            Telangana.add("Jagtial");
            Telangana.add("Jangaon");
            Telangana.add("Jayashankar Bhoopalpally");
            Telangana.add("Jogulamba Gadwal");
            Telangana.add("Kamareddy");
            Telangana.add("Karimnagar");
            Telangana.add("Khammam");
            Telangana.add("Komaram Bheem Asifabad");
            Telangana.add("Mahabubabad");
            Telangana.add("Mahabubnagar");
            Telangana.add("Mancherial");
            Telangana.add("Medak");
            Telangana.add("Medchal");
            Telangana.add("Nagarkurnool");
            Telangana.add("Nalgonda");
            Telangana.add("Nirmal");
            Telangana.add("Nizamabad");
            Telangana.add("Peddapalli");
            Telangana.add("Rajanna Sircilla");
            Telangana.add("Rangareddy");
            Telangana.add("Sangareddy");
            Telangana.add("Siddipet");
            Telangana.add("Suryapet");
            Telangana.add("Vikarabad");
            Telangana.add("Wanaparthy");
            Telangana.add("Warangal (Rural)");
            Telangana.add("Warangal (Urban)");
            Telangana.add("Yadadri Bhuvanagiri");

            Tripura.add("Dhalai");
            Tripura.add("Gomati");
            Tripura.add("Khowai");
            Tripura.add("North Tripura");
            Tripura.add("Sepahijala");
            Tripura.add("South Tripura");
            Tripura.add("Unakoti");
            Tripura.add("West Tripura");

            Uttarakhand.add("Almora");
            Uttarakhand.add("Bageshwar");
            Uttarakhand.add("Chamoli");
            Uttarakhand.add("Champawat");
            Uttarakhand.add("Dehradun");
            Uttarakhand.add("Haridwar");
            Uttarakhand.add("Nainital");
            Uttarakhand.add("Pauri Garhwal");
            Uttarakhand.add("Pithoragarh");
            Uttarakhand.add("Rudraprayag");
            Uttarakhand.add("Tehri Garhwal");
            Uttarakhand.add("Udham Singh Nagar");
            Uttarakhand.add("Uttarkashi");

            UttarPradesh.add("Agra");
            UttarPradesh.add("Aligarh");
            UttarPradesh.add("Allahabad");
            UttarPradesh.add("Ambedkar Nagar");
            UttarPradesh.add("Amethi (Chatrapati Sahuji Mahraj Nagar)");
            UttarPradesh.add("Amroha (J P Nagar)");
            UttarPradesh.add("Auraiya");
            UttarPradesh.add("Azamgarh");
            UttarPradesh.add("Baghpat");
            UttarPradesh.add("Bahraich");
            UttarPradesh.add("Ballia");
            UttarPradesh.add("Balrampur");
            UttarPradesh.add("Banda");
            UttarPradesh.add("Barabanki");
            UttarPradesh.add("Bareilly");
            UttarPradesh.add("Basti");
            UttarPradesh.add("Bhadohi");
            UttarPradesh.add("Bijnor");
            UttarPradesh.add("Budaun");
            UttarPradesh.add("Bulandshahr");
            UttarPradesh.add("Chandauli");
            UttarPradesh.add("Chitrakoot");
            UttarPradesh.add("Deoria");
            UttarPradesh.add("Etah");
            UttarPradesh.add("Etawah");
            UttarPradesh.add("Faizabad");
            UttarPradesh.add("Farrukhabad");
            UttarPradesh.add("Fatehpur");
            UttarPradesh.add("Firozabad");
            UttarPradesh.add("Gautam Buddha Nagar");
            UttarPradesh.add("Ghaziabad");
            UttarPradesh.add("Ghazipur");
            UttarPradesh.add("Gonda");
            UttarPradesh.add("Gorakhpur");
            UttarPradesh.add("Hamirpur");
            UttarPradesh.add("Hapur (Panchsheel Nagar)");
            UttarPradesh.add("Hardoi");
            UttarPradesh.add("Hathras");
            UttarPradesh.add("Jalaun");
            UttarPradesh.add("Jaunpur");
            UttarPradesh.add("Jhansi");
            UttarPradesh.add("Kannauj");
            UttarPradesh.add("Kanpur Dehat");
            UttarPradesh.add("Kanpur Nagar");
            UttarPradesh.add("Kanshiram Nagar (Kasganj)");
            UttarPradesh.add("Kaushambi");
            UttarPradesh.add("Kushinagar (Padrauna)");
            UttarPradesh.add("Lakhimpur - Kheri");
            UttarPradesh.add("Lalitpur");
            UttarPradesh.add("Lucknow");
            UttarPradesh.add("Maharajganj");
            UttarPradesh.add("Mahoba");
            UttarPradesh.add("Mainpuri");
            UttarPradesh.add("Mathura");
            UttarPradesh.add("Mau");
            UttarPradesh.add("Meerut");
            UttarPradesh.add("Mirzapur");
            UttarPradesh.add("Moradabad");
            UttarPradesh.add("Muzaffarnagar");
            UttarPradesh.add("Pilibhit");
            UttarPradesh.add("Pratapgarh");
            UttarPradesh.add("RaeBareli");
            UttarPradesh.add("Rampur");
            UttarPradesh.add("Saharanpur");
            UttarPradesh.add("Sambhal (Bhim Nagar)");
            UttarPradesh.add("Sant Kabir Nagar");
            UttarPradesh.add("Shahjahanpur");
            UttarPradesh.add("Shamali (Prabuddh Nagar)");
            UttarPradesh.add("Shravasti");
            UttarPradesh.add("Siddharth Nagar");
            UttarPradesh.add("Sitapur");
            UttarPradesh.add("Sonbhadra");
            UttarPradesh.add("Sultanpur");
            UttarPradesh.add("Unnao");
            UttarPradesh.add("Varanasi");

            WestBengal.add("Alipurduar");
            WestBengal.add("Bankura");
            WestBengal.add("Birbhum");
            WestBengal.add("Cooch Behar");
            WestBengal.add("Dakshin Dinajpur (South Dinajpur)");
            WestBengal.add("Darjeeling");
            WestBengal.add("Hooghly");
            WestBengal.add("Howrah");
            WestBengal.add("Jalpaiguri");
            WestBengal.add("Jhargram");
            WestBengal.add("Kalimpong");
            WestBengal.add("Kolkata");
            WestBengal.add("Malda");
            WestBengal.add("Murshidabad");
            WestBengal.add("Nadia");
            WestBengal.add("North 24 Parganas");
            WestBengal.add("Paschim Medinipur (West Medinipur)");
            WestBengal.add("Paschim (West) Burdwan (Bardhaman)");
            WestBengal.add("Purba Burdwan (Bardhaman)");
            WestBengal.add("Purba Medinipur (East Medinipur)");
            WestBengal.add("Purulia");
            WestBengal.add("South 24 Parganas");
            WestBengal.add("Uttar Dinajpur (North Dinajpur)");

        }

        spinner.setOnItemSelectedListener(new StateUTSpinnerClass());
        spinner2.setOnItemSelectedListener(new LocalityUTSpinnerClass());

        adapter = new ArrayAdapter(this, android.R.layout.simple_spinner_item, list);

        spinner.setAdapter(adapter);


        String localArea = list.get(spinner.getSelectedItemPosition());


        if (localArea.equals("Andaman and Nicobar Islands")){mainList = AndamanandNicobarIsland;}
        else if (localArea.equals("Andhra Pradesh")){mainList = AndhraPradesh;}
        else if (localArea.equals("Arunachal Pradesh")){mainList = ArunachalPradesh;}
        else if (localArea.equals("Assam")){mainList = Assam;}
        else if (localArea.equals("Bihar")){mainList = Bihar;}
        else if (localArea.equals("Chandigarh")){mainList = Chandigarh;}
        else if (localArea.equals("Chhattisgarh")){mainList = Chhattisgarh;}
        else if (localArea.equals("Dadra and Nagar Haveli")){mainList = DadraandNagarHaveli;}
        else if (localArea.equals("Daman and Diu")){mainList = DamanandDiu;}
        else if (localArea.equals("Delhi")){mainList = Delhi;}
        else if (localArea.equals("Goa")){mainList = Goa;}
        else if (localArea.equals("Gujarat")){mainList = Gujarat;}
        else if (localArea.equals("Haryana")){mainList = Haryana;}
        else if (localArea.equals("Himachal Pradesh")){mainList = HimachalPradesh;}
        else if (localArea.equals("Jammu and Kashmir")){mainList = JammuandKashmir;}
        else if (localArea.equals("Jharkhand")){mainList = Jharkhand;}
        else if (localArea.equals("Karnataka")){mainList = Karnataka;}
        else if (localArea.equals("Kerala")){mainList = Kerala;}
        else if (localArea.equals("Ladakh")){mainList = Ladakh;}
        else if (localArea.equals("Lakshadweep")){mainList = Lakshadweep;}
        else if (localArea.equals("Madhya Pradesh")){mainList = MadhyaPradesh;}
        else if (localArea.equals("Maharashtra")){mainList = Maharashtra;}
        else if (localArea.equals("Manipur")){mainList = Manipur;}
        else if (localArea.equals("Meghalaya")){mainList = Meghalaya;}
        else if (localArea.equals("Mizoram")){mainList = Mizoram;}
        else if (localArea.equals("Nagaland")){mainList = Nagaland;}
        else if (localArea.equals("Odisha")){mainList = Odisha;}
        else if (localArea.equals("Puducherry")){mainList = Puducherry;}
        else if (localArea.equals("Punjab")){mainList = Punjab;}
        else if (localArea.equals("Rajasthan")){mainList = Rajasthan;}
        else if (localArea.equals("Sikkim")){mainList = Sikkim;}
        else if (localArea.equals("Tamil Nadu")){mainList = TamilNadu;}
        else if (localArea.equals("Telangana")){mainList = Telangana;}
        else if (localArea.equals("Tripura")){mainList = Tripura;}
        else if (localArea.equals("Uttar Pradesh")){mainList = UttarPradesh;}
        else if (localArea.equals("Uttarakhand")){mainList = Uttarakhand;}
        else if (localArea.equals("West Bengal")){mainList = WestBengal;}


       Button button = findViewById(R.id.continueFromLocation);




        adapter2 = new ArrayAdapter(this, android.R.layout.simple_spinner_item, mainList);
        spinner2.setAdapter(adapter2);
        adapter2.notifyDataSetChanged();



        Intent intent = new Intent(UploadPageOne.this, ShowSingleStateAndData.class);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                StateUT = list.get(spinner.getSelectedItemPosition());
                locality = mainList.get(spinner2.getSelectedItemPosition());

                intent.putExtra("StateUT", StateUT);
                intent.putExtra("District", locality);
                startActivity(intent);
            }
        });




    }



    class StateUTSpinnerClass implements AdapterView.OnItemSelectedListener {

        @Override
        public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {


            String localArea = list.get(position);


            if (localArea.equals("Andaman and Nicobar Islands")){
                mainList = AndamanandNicobarIsland;}
            else if (localArea.equals("Andhra Pradesh")){

                mainList = AndhraPradesh;
            }
            else if (localArea.equals("Arunachal Pradesh")){mainList = ArunachalPradesh;}
            else if (localArea.equals("Assam")){mainList = Assam;}
            else if (localArea.equals("Bihar")){mainList = Bihar;}
            else if (localArea.equals("Chandigarh")){mainList = Chandigarh;}
            else if (localArea.equals("Chhattisgarh")){mainList = Chhattisgarh;}
            else if (localArea.equals("Dadra and Nagar Haveli")){mainList = DadraandNagarHaveli;}
            else if (localArea.equals("Daman and Diu")){mainList = DamanandDiu;}
            else if (localArea.equals("Delhi")){mainList = Delhi;}
            else if (localArea.equals("Goa")){mainList = Goa;}
            else if (localArea.equals("Gujarat")){mainList = Gujarat;}
            else if (localArea.equals("Haryana")){mainList = Haryana;}
            else if (localArea.equals("Himachal Pradesh")){mainList = HimachalPradesh;}
            else if (localArea.equals("Jammu and Kashmir")){mainList = JammuandKashmir;}
            else if (localArea.equals("Jharkhand")){mainList = Jharkhand;}
            else if (localArea.equals("Karnataka")){mainList = Karnataka;}
            else if (localArea.equals("Kerala")){mainList = Kerala;}
            else if (localArea.equals("Ladakh")){mainList = Ladakh;}
            else if (localArea.equals("Lakshadweep")){mainList = Lakshadweep;}
            else if (localArea.equals("Madhya Pradesh")){mainList = MadhyaPradesh;}
            else if (localArea.equals("Maharashtra")){mainList = Maharashtra;}
            else if (localArea.equals("Manipur")){mainList = Manipur;}
            else if (localArea.equals("Meghalaya")){mainList = Meghalaya;}
            else if (localArea.equals("Mizoram")){mainList = Mizoram;}
            else if (localArea.equals("Nagaland")){mainList = Nagaland;}
            else if (localArea.equals("Odisha")){mainList = Odisha;}
            else if (localArea.equals("Puducherry")){mainList = Puducherry;}
            else if (localArea.equals("Punjab")){mainList = Punjab;}
            else if (localArea.equals("Rajasthan")){mainList = Rajasthan;}
            else if (localArea.equals("Sikkim")){mainList = Sikkim;}
            else if (localArea.equals("Tamil Nadu")){mainList = TamilNadu;}
            else if (localArea.equals("Telangana")){mainList = Telangana;}
            else if (localArea.equals("Tripura")){mainList = Tripura;}
            else if (localArea.equals("Uttar Pradesh")){mainList = UttarPradesh;}
            else if (localArea.equals("Uttarakhand")){mainList = Uttarakhand;}
            else if (localArea.equals("West Bengal")){mainList = WestBengal;}


            adapter2 = new ArrayAdapter(UploadPageOne.this, android.R.layout.simple_spinner_item, mainList);
            spinner2.setAdapter(adapter2);
            adapter2.notifyDataSetChanged();


        }

        @Override
        public void onNothingSelected(AdapterView<?> parent) {

        }
    }

    class LocalityUTSpinnerClass implements AdapterView.OnItemSelectedListener {

        @Override
        public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {

        }

        @Override
        public void onNothingSelected(AdapterView<?> parent) {

        }
    }



}
