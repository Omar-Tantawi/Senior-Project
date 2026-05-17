<?php

namespace App\Http\Requests\Submission;

use Illuminate\Foundation\Http\FormRequest;

class StoreSubmissionRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'homework_id' => 'required|integer',
            'file'        => 'nullable|file|max:10240|mimes:pdf,doc,docx,ppt,pptx,xls,xlsx,txt,zip,jpg,jpeg,png',
        ];
    }
}
